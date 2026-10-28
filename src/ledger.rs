//! Banking ledger and concurrent account transfers.
//!
//! Port of Go `pkg/ledger`. Balances are `i64` in the smallest
//! currency unit (cents), exactly as in Go.
//!
//! Concurrency traps preserved:
//!
//! * Deadlock prevention via deterministic lock ordering: transfers
//!   acquire per-account locks in lexicographic account-ID order, so
//!   `A -> B` and `B -> A` can never hold locks in opposite orders.
//! * Balance invariants are checked while holding the account lock,
//!   before any mutation.
//! * Multi-leg transfers pre-validate every leg's net delta before
//!   committing anything; on failure nothing moves and the record is
//!   `RolledBack`.
//! * The transaction history is append-only.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

/// Errors a ledger operation can report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    NegativeAmount,
    InsufficientFunds { balance: i64, amount: i64 },
    AccountNotFound,
    AccountExists,
    SelfTransfer,
    InvalidTransaction(String),
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LedgerError::NegativeAmount => write!(f, "amount must be strictly positive"),
            LedgerError::InsufficientFunds { balance, amount } => {
                write!(
                    f,
                    "insufficient funds for withdrawal: balance {balance} < {amount}"
                )
            }
            LedgerError::AccountNotFound => write!(f, "account not found"),
            LedgerError::AccountExists => write!(f, "account already exists"),
            LedgerError::SelfTransfer => write!(f, "cannot transfer to the same account"),
            LedgerError::InvalidTransaction(msg) => {
                write!(f, "invalid transaction specification: {msg}")
            }
        }
    }
}

impl std::error::Error for LedgerError {}

/// A single banking account. Fine-grained per-account locking avoids a
/// single global ledger bottleneck, as in Go.
#[derive(Debug)]
pub struct Account {
    id: String,
    balance: RwLock<i64>,
}

impl Account {
    /// Create an account. Negative initial balances are rejected.
    pub fn new(id: &str, initial_balance: i64) -> Result<Arc<Account>, LedgerError> {
        if initial_balance < 0 {
            return Err(LedgerError::NegativeAmount);
        }
        Ok(Arc::new(Account {
            id: id.to_string(),
            balance: RwLock::new(initial_balance),
        }))
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// Current balance, read under a shared lock.
    pub fn balance(&self) -> i64 {
        *self.balance.read().unwrap_or_else(|e| e.into_inner())
    }
}

/// Classification of a financial operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxType {
    Deposit,
    Withdrawal,
    Transfer,
    MultiTransfer,
}

impl fmt::Display for TxType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TxType::Deposit => "DEPOSIT",
            TxType::Withdrawal => "WITHDRAWAL",
            TxType::Transfer => "TRANSFER",
            TxType::MultiTransfer => "MULTI_TRANSFER",
        };
        write!(f, "{s}")
    }
}

/// Settlement state of a transaction record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxStatus {
    Committed,
    RolledBack,
    Failed,
}

impl fmt::Display for TxStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TxStatus::Committed => "COMMITTED",
            TxStatus::RolledBack => "ROLLED_BACK",
            TxStatus::Failed => "FAILED",
        };
        write!(f, "{s}")
    }
}

/// One leg of a transfer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TxLeg {
    pub from_account: String,
    pub to_account: String,
    pub amount: i64,
}

impl TxLeg {
    pub fn new(from: &str, to: &str, amount: i64) -> TxLeg {
        TxLeg {
            from_account: from.to_string(),
            to_account: to.to_string(),
            amount,
        }
    }
}

/// An immutable audit-log entry.
///
/// Go returns a `*TransactionRecord` alongside an error for failed
/// operations, so callers can inspect the record even on failure. Rust
/// has no nil pointer for that, so operations that can fail return a
/// [`TxOutcome`]: the record plus an optional error.
#[derive(Clone, Debug)]
pub struct TransactionRecord {
    pub id: String,
    pub tx_type: TxType,
    pub status: TxStatus,
    pub legs: Vec<TxLeg>,
    pub timestamp: SystemTime,
    pub error: Option<String>,
}

/// The result of a ledger operation: always a record, plus an error
/// when the operation did not commit.
#[derive(Clone, Debug)]
pub struct TxOutcome {
    pub record: TransactionRecord,
    pub error: Option<LedgerError>,
}

impl TxOutcome {
    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }
}

/// The ledger: accounts plus an append-only transaction history.
#[derive(Debug)]
pub struct Ledger {
    accounts: RwLock<HashMap<String, Arc<Account>>>,
    history: RwLock<Vec<TransactionRecord>>,
}

static TX_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Go mints IDs from 8 crypto-random bytes. The standard library has
/// no RNG, so IDs here combine a process-wide counter with the current
/// time in nanoseconds, hex-encoded — unique within a process, which
/// is all an audit-log ID needs.
fn new_id() -> String {
    let n = TX_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{now:x}-{n:x}")
}

impl Ledger {
    pub fn new() -> Ledger {
        Ledger::default()
    }

    /// Register a new account. Duplicate IDs are rejected.
    pub fn create_account(
        &self,
        id: &str,
        initial_balance: i64,
    ) -> Result<Arc<Account>, LedgerError> {
        let mut accounts = self.accounts.write().unwrap_or_else(|e| e.into_inner());
        if accounts.contains_key(id) {
            return Err(LedgerError::AccountExists);
        }
        let acc = Account::new(id, initial_balance)?;
        accounts.insert(id.to_string(), Arc::clone(&acc));
        // Record the opening deposit after releasing nothing: the
        // history lock is independent of the accounts lock, and every
        // path in this file takes accounts-then-history, never the
        // reverse, so no lock-order cycle exists. (Go holds the
        // accounts lock across this call; holding it here too would
        // be equivalent, but recording after the insert keeps the
        // critical section shorter. The deposit record itself is
        // appended under the history lock only.)
        drop(accounts);
        if initial_balance > 0 {
            self.record(TransactionRecord {
                id: new_id(),
                tx_type: TxType::Deposit,
                status: TxStatus::Committed,
                legs: vec![TxLeg {
                    from_account: String::new(),
                    to_account: id.to_string(),
                    amount: initial_balance,
                }],
                timestamp: SystemTime::now(),
                error: None,
            });
        }
        Ok(acc)
    }

    pub fn get_account(&self, id: &str) -> Result<Arc<Account>, LedgerError> {
        self.accounts
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
            .cloned()
            .ok_or(LedgerError::AccountNotFound)
    }

    /// Credit an account directly.
    pub fn deposit(&self, account_id: &str, amount: i64) -> TxOutcome {
        let mut record = TransactionRecord {
            id: new_id(),
            tx_type: TxType::Deposit,
            status: TxStatus::Committed,
            legs: vec![TxLeg {
                from_account: String::new(),
                to_account: account_id.to_string(),
                amount,
            }],
            timestamp: SystemTime::now(),
            error: None,
        };
        let error = match self.get_account(account_id) {
            Err(e) => Some(e),
            Ok(acc) => {
                let mut balance = acc.balance.write().unwrap_or_else(|e| e.into_inner());
                if amount <= 0 {
                    Some(LedgerError::NegativeAmount)
                } else {
                    *balance += amount;
                    None
                }
            }
        };
        if let Some(e) = &error {
            record.status = TxStatus::Failed;
            record.error = Some(e.to_string());
        }
        self.record(record.clone());
        TxOutcome { record, error }
    }

    /// Debit an account directly. Fails instead of going negative.
    pub fn withdraw(&self, account_id: &str, amount: i64) -> TxOutcome {
        let mut record = TransactionRecord {
            id: new_id(),
            tx_type: TxType::Withdrawal,
            status: TxStatus::Committed,
            legs: vec![TxLeg {
                from_account: account_id.to_string(),
                to_account: String::new(),
                amount,
            }],
            timestamp: SystemTime::now(),
            error: None,
        };
        let error = match self.get_account(account_id) {
            Err(e) => Some(e),
            Ok(acc) => {
                let mut balance = acc.balance.write().unwrap_or_else(|e| e.into_inner());
                if amount <= 0 {
                    Some(LedgerError::NegativeAmount)
                } else if *balance < amount {
                    Some(LedgerError::InsufficientFunds {
                        balance: *balance,
                        amount,
                    })
                } else {
                    *balance -= amount;
                    None
                }
            }
        };
        if let Some(e) = &error {
            record.status = TxStatus::Failed;
            record.error = Some(e.to_string());
        }
        self.record(record.clone());
        TxOutcome { record, error }
    }

    /// Atomically transfer funds from one account to another.
    ///
    /// Locks are acquired in lexicographic account-ID order, so two
    /// opposite transfers can never deadlock waiting on each other.
    pub fn transfer(&self, from_id: &str, to_id: &str, amount: i64) -> TxOutcome {
        let mut record = TransactionRecord {
            id: new_id(),
            tx_type: TxType::Transfer,
            status: TxStatus::Committed,
            legs: vec![TxLeg::new(from_id, to_id, amount)],
            timestamp: SystemTime::now(),
            error: None,
        };
        // Failures that Go returns as (nil, err) with no record are
        // returned here as a record with status Failed, so the caller
        // always has the audit entry. The error itself is unchanged.
        let early_error = if from_id == to_id {
            Some(LedgerError::SelfTransfer)
        } else if amount <= 0 {
            Some(LedgerError::NegativeAmount)
        } else {
            None
        };
        if let Some(e) = early_error {
            record.status = TxStatus::Failed;
            record.error = Some(e.to_string());
            self.record(record.clone());
            return TxOutcome {
                record,
                error: Some(e),
            };
        }
        let (from_acc, to_acc) = match (self.get_account(from_id), self.get_account(to_id)) {
            (Ok(a), Ok(b)) => (a, b),
            _ => {
                record.status = TxStatus::Failed;
                record.error = Some(LedgerError::AccountNotFound.to_string());
                self.record(record.clone());
                return TxOutcome {
                    record,
                    error: Some(LedgerError::AccountNotFound),
                };
            }
        };
        // Deterministic lock ordering.
        let (first, second) = if from_id <= to_id {
            (Arc::clone(&from_acc), Arc::clone(&to_acc))
        } else {
            (Arc::clone(&to_acc), Arc::clone(&from_acc))
        };
        let mut first_guard = first.balance.write().unwrap_or_else(|e| e.into_inner());
        let mut second_guard = second.balance.write().unwrap_or_else(|e| e.into_inner());
        // Which guard is the source depends on the sort order, so
        // branch on it instead of aliasing both guards at once (the
        // borrow checker forbids holding two mutable borrows of the
        // same guard, which aliasing would do).
        let error = if from_id <= to_id {
            if *first_guard < amount {
                Some(LedgerError::InsufficientFunds {
                    balance: *first_guard,
                    amount,
                })
            } else {
                *first_guard -= amount;
                *second_guard += amount;
                None
            }
        } else if *second_guard < amount {
            Some(LedgerError::InsufficientFunds {
                balance: *second_guard,
                amount,
            })
        } else {
            *second_guard -= amount;
            *first_guard += amount;
            None
        };
        drop(second_guard);
        drop(first_guard);
        if let Some(error) = error {
            record.status = TxStatus::RolledBack;
            record.error = Some(error.to_string());
            self.record(record.clone());
            return TxOutcome {
                record,
                error: Some(error),
            };
        }
        self.record(record.clone());
        TxOutcome {
            record,
            error: None,
        }
    }

    /// Atomically transfer across an arbitrary number of accounts.
    ///
    /// All unique account IDs are locked in sorted order, every net
    /// delta is validated before any balance moves, and on failure
    /// nothing moves at all.
    pub fn multi_transfer(&self, legs: &[TxLeg]) -> TxOutcome {
        let mut record = TransactionRecord {
            id: new_id(),
            tx_type: TxType::MultiTransfer,
            status: TxStatus::Committed,
            legs: legs.to_vec(),
            timestamp: SystemTime::now(),
            error: None,
        };
        if legs.is_empty() {
            let e = LedgerError::InvalidTransaction("no legs".to_string());
            record.status = TxStatus::Failed;
            record.error = Some(e.to_string());
            self.record(record.clone());
            return TxOutcome {
                record,
                error: Some(e),
            };
        }
        let mut account_ids: HashSet<String> = HashSet::new();
        let mut net_deltas: HashMap<String, i64> = HashMap::new();
        for leg in legs {
            if leg.amount <= 0 {
                let e = LedgerError::NegativeAmount;
                record.status = TxStatus::Failed;
                record.error = Some("transfer leg amount must be positive".to_string());
                self.record(record.clone());
                return TxOutcome {
                    record,
                    error: Some(e),
                };
            }
            if leg.from_account == leg.to_account {
                let e = LedgerError::SelfTransfer;
                record.status = TxStatus::Failed;
                record.error = Some("cannot transfer to the same account".to_string());
                self.record(record.clone());
                return TxOutcome {
                    record,
                    error: Some(e),
                };
            }
            account_ids.insert(leg.from_account.clone());
            account_ids.insert(leg.to_account.clone());
            *net_deltas.entry(leg.from_account.clone()).or_insert(0) -= leg.amount;
            *net_deltas.entry(leg.to_account.clone()).or_insert(0) += leg.amount;
        }
        // Fetch all accounts (accounts lock released before any
        // per-account lock is taken, as in Go).
        let accounts_snapshot = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        let mut account_map: HashMap<String, Arc<Account>> = HashMap::new();
        for id in &account_ids {
            match accounts_snapshot.get(id) {
                Some(acc) => {
                    account_map.insert(id.clone(), Arc::clone(acc));
                }
                None => {
                    let e = LedgerError::AccountNotFound;
                    record.status = TxStatus::Failed;
                    record.error = Some(format!("account {id} not found"));
                    drop(accounts_snapshot);
                    self.record(record.clone());
                    return TxOutcome {
                        record,
                        error: Some(e),
                    };
                }
            }
        }
        drop(accounts_snapshot);
        let mut sorted_ids: Vec<String> = account_ids.into_iter().collect();
        sorted_ids.sort();
        // Acquire write locks in sorted order and hold them all.
        let mut guards: Vec<std::sync::RwLockWriteGuard<'_, i64>> =
            Vec::with_capacity(sorted_ids.len());
        for id in &sorted_ids {
            guards.push(
                account_map[id]
                    .balance
                    .write()
                    .unwrap_or_else(|e| e.into_inner()),
            );
        }
        // Validate every net delta before committing anything.
        for (id, delta) in &net_deltas {
            let idx = sorted_ids.iter().position(|x| x == id).unwrap();
            if *guards[idx] + *delta < 0 {
                let error = LedgerError::InsufficientFunds {
                    balance: *guards[idx],
                    amount: -*delta,
                };
                record.status = TxStatus::RolledBack;
                record.error = Some(format!(
                    "insufficient funds for account {id}: balance {} + delta {delta} < 0",
                    *guards[idx]
                ));
                drop(guards);
                self.record(record.clone());
                return TxOutcome {
                    record,
                    error: Some(error),
                };
            }
        }
        for (id, delta) in &net_deltas {
            let idx = sorted_ids.iter().position(|x| x == id).unwrap();
            *guards[idx] += *delta;
        }
        drop(guards);
        self.record(record.clone());
        TxOutcome {
            record,
            error: None,
        }
    }

    /// Total funds across all accounts, from a consistent snapshot:
    /// every account is read-locked in sorted order while summing.
    pub fn total_system_balance(&self) -> i64 {
        let snapshot = self.accounts.read().unwrap_or_else(|e| e.into_inner());
        let mut sorted_ids: Vec<String> = snapshot.keys().cloned().collect();
        sorted_ids.sort();
        let accounts: Vec<Arc<Account>> = sorted_ids
            .iter()
            .map(|id| Arc::clone(&snapshot[id]))
            .collect();
        drop(snapshot);
        let guards: Vec<std::sync::RwLockReadGuard<'_, i64>> = accounts
            .iter()
            .map(|a| a.balance.read().unwrap_or_else(|e| e.into_inner()))
            .collect();
        guards.iter().map(|g| **g).sum()
    }

    /// A copy of the append-only audit log.
    pub fn history(&self) -> Vec<TransactionRecord> {
        self.history
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn record(&self, tx: TransactionRecord) {
        self.history
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push(tx);
    }
}

impl Default for Ledger {
    fn default() -> Self {
        Ledger {
            accounts: RwLock::new(HashMap::new()),
            history: RwLock::new(Vec::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicI64;
    use std::thread;

    fn seeded_ledger() -> Ledger {
        Ledger::new()
    }

    #[test]
    fn deadlock_freedom_bidirectional_transfers() {
        let ledger = Arc::new(seeded_ledger());
        let a = ledger.create_account("acc-A", 100_000).unwrap();
        let b = ledger.create_account("acc-B", 100_000).unwrap();
        let initial = a.balance() + b.balance();
        let l1 = Arc::clone(&ledger);
        let l2 = Arc::clone(&ledger);
        let h1 = thread::spawn(move || {
            for _ in 0..500 {
                let _ = l1.transfer("acc-A", "acc-B", 10);
            }
        });
        let h2 = thread::spawn(move || {
            for _ in 0..500 {
                let _ = l2.transfer("acc-B", "acc-A", 10);
            }
        });
        h1.join().unwrap();
        h2.join().unwrap();
        assert_eq!(a.balance() + b.balance(), initial);
    }

    #[test]
    fn balance_invariant_no_double_spend() {
        let ledger = Arc::new(seeded_ledger());
        let acc = ledger.create_account("target-acc", 100).unwrap();
        let ok = AtomicI64::new(0);
        let failed = AtomicI64::new(0);
        thread::scope(|s| {
            for _ in 0..20 {
                s.spawn(|| {
                    let outcome = ledger.withdraw("target-acc", 100);
                    if outcome.is_ok() {
                        ok.fetch_add(1, Ordering::SeqCst);
                    } else {
                        failed.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
        });
        assert_eq!(ok.load(Ordering::SeqCst), 1);
        assert_eq!(failed.load(Ordering::SeqCst), 19);
        assert_eq!(acc.balance(), 0);
    }

    #[test]
    fn multi_transfer_atomic_rollback() {
        let ledger = seeded_ledger();
        let a = ledger.create_account("acc-A", 100).unwrap();
        let b = ledger.create_account("acc-B", 50).unwrap();
        let c = ledger.create_account("acc-C", 50).unwrap();
        let outcome = ledger.multi_transfer(&[
            TxLeg::new("acc-A", "acc-B", 60),
            TxLeg::new("acc-A", "acc-C", 60),
        ]);
        assert!(outcome.error.is_some());
        assert_eq!(outcome.record.status, TxStatus::RolledBack);
        assert_eq!(a.balance(), 100);
        assert_eq!(b.balance(), 50);
        assert_eq!(c.balance(), 50);
    }

    #[test]
    fn multi_party_transfer_success() {
        let ledger = seeded_ledger();
        let a = ledger.create_account("acc-A", 200).unwrap();
        let b = ledger.create_account("acc-B", 50).unwrap();
        let c = ledger.create_account("acc-C", 50).unwrap();
        let outcome = ledger.multi_transfer(&[
            TxLeg::new("acc-A", "acc-B", 60),
            TxLeg::new("acc-A", "acc-C", 40),
        ]);
        assert!(outcome.error.is_none());
        assert_eq!(outcome.record.status, TxStatus::Committed);
        assert_eq!(a.balance(), 100);
        assert_eq!(b.balance(), 110);
        assert_eq!(c.balance(), 90);
    }

    #[test]
    fn high_concurrent_n_party_transfers_total_conservation() {
        let ledger = Arc::new(seeded_ledger());
        for i in 0..10 {
            ledger
                .create_account(&format!("user-{i:02}"), 10_000)
                .unwrap();
        }
        thread::scope(|s| {
            for w in 0..30 {
                let ledger = Arc::clone(&ledger);
                s.spawn(move || {
                    // A tiny xorshift PRNG, seeded per worker, in place
                    // of Go's math/rand with a per-worker seed.
                    let mut state: u64 = (w as u64 + 100).wrapping_mul(0x9E3779B97F4A7C15);
                    let mut next = move || {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        state
                    };
                    for _ in 0..100 {
                        let from = (next() % 10) as usize;
                        let to = (next() % 10) as usize;
                        if from == to {
                            continue;
                        }
                        let amount = (next() % 50 + 1) as i64;
                        let _ = ledger.transfer(
                            &format!("user-{from:02}"),
                            &format!("user-{to:02}"),
                            amount,
                        );
                    }
                });
            }
        });
        assert_eq!(ledger.total_system_balance(), 100_000);
    }
}
