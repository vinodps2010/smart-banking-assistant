-- ============================================================
-- Smart Banking Assistant
-- Transactional RDBMS Schema
-- PostgreSQL 16+
-- ============================================================

-- ============================================================
-- 0. EXTENSIONS
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ============================================================
-- 1. ACCOUNTS
-- ============================================================

CREATE TABLE IF NOT EXISTS accounts (
    account_id VARCHAR(20) PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    account_type VARCHAR(20) NOT NULL
        CHECK (
            account_type IN (
                'savings',
                'current',
                'salary'
            )
        ),
    branch_code VARCHAR(10) NOT NULL,
    ifsc_code VARCHAR(15),
    mobile VARCHAR(15),
    email VARCHAR(100),
    kyc_status VARCHAR(20) DEFAULT 'verified',
    created_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 2. TRANSACTION HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS transactions (
    txn_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(20)
        REFERENCES accounts(account_id),
    txn_date DATE NOT NULL,
    txn_type VARCHAR(10) NOT NULL
        CHECK (
            txn_type IN (
                'debit',
                'credit'
            )
        ),
    amount NUMERIC(15,2) NOT NULL,
    balance_after NUMERIC(15,2),
    description VARCHAR(200),
    channel VARCHAR(20)
        CHECK (
            channel IN (
                'ATM',
                'UPI',
                'NEFT',
                'RTGS',
                'IMPS',
                'branch',
                'online',
                'POS'
            )
        ),
    merchant_name VARCHAR(100),
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 3. LOAN ACCOUNTS
-- ============================================================

CREATE TABLE IF NOT EXISTS loan_accounts (
    loan_id VARCHAR(20) PRIMARY KEY,
    account_id VARCHAR(20)
        REFERENCES accounts(account_id),
    loan_type VARCHAR(30) NOT NULL
        CHECK (
            loan_type IN (
                'home_loan',
                'personal_loan',
                'auto_loan',
                'gold_loan'
            )
        ),
    principal NUMERIC(15,2) NOT NULL,
    outstanding NUMERIC(15,2) NOT NULL,
    disbursed_date DATE,
    emi_amount NUMERIC(15,2),
    next_emi_date DATE,
    interest_rate NUMERIC(5,2),
    tenure_months INT,
    emi_paid INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 4. FIXED DEPOSITS
-- ============================================================

CREATE TABLE IF NOT EXISTS fixed_deposits (
    fd_id VARCHAR(20) PRIMARY KEY,
    account_id VARCHAR(20)
        REFERENCES accounts(account_id),
    principal NUMERIC(15,2) NOT NULL,
    interest_rate NUMERIC(5,2) NOT NULL,
    tenure_days INT NOT NULL,
    start_date DATE NOT NULL,
    maturity_date DATE NOT NULL,
    maturity_amount NUMERIC(15,2),
    interest_payout VARCHAR(20) DEFAULT 'at_maturity',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 5. CREDIT CARDS
-- ============================================================

CREATE TABLE IF NOT EXISTS credit_cards (
    card_id VARCHAR(20) PRIMARY KEY,
    account_id VARCHAR(20)
        REFERENCES accounts(account_id),
    card_variant VARCHAR(30),
    credit_limit NUMERIC(15,2),
    available_limit NUMERIC(15,2),
    outstanding_amt NUMERIC(15,2) DEFAULT 0,
    due_date DATE,
    min_due NUMERIC(15,2) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active',
    issued_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);


-- ============================================================
-- 6. CREDIT CARD TRANSACTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS card_transactions (
    txn_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    card_id VARCHAR(20)
        REFERENCES credit_cards(card_id),
    txn_date DATE NOT NULL,
    txn_type VARCHAR(20)
        CHECK (
            txn_type IN (
                'purchase',
                'cashadvance',
                'payment',
                'refund',
                'fee'
            )
        ),
    amount NUMERIC(15,2) NOT NULL,
    merchant_name VARCHAR(100),
    category VARCHAR(50),
    is_international BOOLEAN DEFAULT FALSE,
    currency VARCHAR(5) DEFAULT 'INR',
    created_at TIMESTAMP DEFAULT NOW()
);
