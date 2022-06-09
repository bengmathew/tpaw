use crate::params::*;
use crate::pre_calculations::PreCalculations;
use crate::utils::*;
use serde::{Deserialize, Serialize};

pub struct SingleYearContext<'a> {
    pub params: &'a Params,
    pub pre_calculations: &'a PreCalculations,
    pub year_index: usize,
    pub returns: &'a ReturnsAtPointInTime,
    pub balance_starting: f64,
}

#[derive(Serialize, Deserialize)]
pub struct AfterContributions {
    pub contributions: f64,
    pub balance: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Withdrawals {
    pub essential: f64,
    pub discretionary: f64,
    pub regular: f64,
    pub total: f64,
    pub from_savings_portfolio_rate: f64,
}

#[derive(Serialize, Deserialize)]
pub struct AfterWithdrawals {
    pub contributions: f64,
    pub withdrawals: Withdrawals,
    pub balance: f64,
}

#[derive(Serialize, Deserialize)]
pub struct End {
    pub stock_allocation: f64,
    pub balance: f64,
}

pub struct TargetWithdrawals {
    pub lmp: f64,
    pub essential: f64,
    pub discretionary: f64,
    pub regular_without_lmp: f64,
}

pub struct ActualWithdrawals {
    pub essential: f64,
    pub discretionary: f64,
    pub regular: f64,
}

#[inline(always)]
pub fn apply_contributions(contributions: f64, balance_starting: f64) -> AfterContributions {
    let balance = balance_starting + contributions;
    AfterContributions {
        contributions,
        balance,
    }
}

#[inline(always)]
fn get_actual_withdrawals(
    target: &TargetWithdrawals,
    context: &SingleYearContext,
    after_contributions: &AfterContributions,
) -> ActualWithdrawals {
    // ---- Apply ceiling and floor ----
    let target = {
        let params = context.params;
        let year_index = context.year_index;
        let mut discretionary = target.discretionary;

        let withdrawal_started = (year_index as i32) >= (params.withdrawal_start_year);
        let mut regular_with_lmp = target.lmp + target.regular_without_lmp;

        if let Some(spending_ceiling) = params.spending_ceiling {
            discretionary = f64::min(
                discretionary,
                params.by_year.withdrawals_discretionary[year_index],
            );
            regular_with_lmp = f64::min(regular_with_lmp, spending_ceiling);
        };

        if let Some(spending_floor) = params.spending_floor {
            discretionary = f64::max(
                discretionary,
                params.by_year.withdrawals_discretionary[year_index],
            );
            if withdrawal_started {
                regular_with_lmp = f64::max(regular_with_lmp, spending_floor);
            };
        }
        let regular_without_lmp = regular_with_lmp - target.lmp;

        // assert!(regular_without_lmp >= 0.0);
        TargetWithdrawals {
            lmp: target.lmp,
            essential: target.essential,
            discretionary,
            regular_without_lmp,
        }
    };

    // Apply balance constraints.
    let mut account = AccountForWithdrawal {
        balance: after_contributions.balance,
    };
    let lmp = account.withdraw(target.lmp);
    ActualWithdrawals {
        essential: account.withdraw(target.essential),
        discretionary: account.withdraw(target.discretionary),
        regular: lmp + account.withdraw(target.regular_without_lmp),
    }
}

#[inline(always)]
pub fn apply_target_withdrawals(
    target: &TargetWithdrawals,
    context: &SingleYearContext,
    after_contributions: &AfterContributions,
) -> AfterWithdrawals {
    let balance_starting = context.balance_starting;
    let withdrawals = get_actual_withdrawals(target, context, after_contributions);
    let ActualWithdrawals {
        essential,
        discretionary,
        regular,
    } = withdrawals;
    let balance_starting = balance_starting;
    let contributions = after_contributions.contributions;

    let total = { essential + discretionary + regular };

    // assert!(total <= after_contributions.balance);

    let balance = after_contributions.balance - total;
    let from_contributions = f64::min(total, contributions);
    let from_savings_portfolio = total - from_contributions;
    let from_savings_portfolio_rate = if balance_starting == 0.0 {
        0.0
    } else {
        from_savings_portfolio / balance_starting
    };

    AfterWithdrawals {
        contributions,
        withdrawals: Withdrawals {
            essential,
            discretionary,
            regular,
            total,
            from_savings_portfolio_rate,
        },
        balance,
    }
}

#[inline(always)]
pub fn apply_allocation(
    stock_allocation: f64,
    return_rate: &ReturnsAtPointInTime,
    after_withdrawals: &AfterWithdrawals,
) -> End {
    // Don't use blend_returns here. It is extremely slow.
    let rate = return_rate.bonds * (1.0 - stock_allocation) + return_rate.stocks * stock_allocation;
    let balance = after_withdrawals.balance * (1.0 + rate);
    End {
        stock_allocation,
        balance,
    }
}
