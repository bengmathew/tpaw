#![allow(non_camel_case_types)]

use crate::{
    market_data::market_data_defs::VTAndBNDData_PercentageChangeFromLastClose, to_float_wire::ToFloatWire, wire::{
        wire_portfolio_balance_estimation_action,
        wire_portfolio_balance_estimation_action_args_withdrawal_and_contribution,
        WirePortfolioBalanceEstimationAction, WirePortfolioBalanceEstimationActionArgsMarketClose,
        WirePortfolioBalanceEstimationActionArgsMonthlyRebalance,
        WirePortfolioBalanceEstimationActionArgsPlanChange,
        WirePortfolioBalanceEstimationActionArgsPlanChangePortfolioUpdate,
        WirePortfolioBalanceEstimationActionArgsWithdrawalAndContribution,
        WirePortfolioBalanceEstimationResult, WirePortfolioBalanceEstimationState,
        WirePortfolioBalanceEstimationStateChange, WireStockAllocation,
        WireVtAndBndPercentageChange,
    }
};

#[derive(Clone)]
pub struct PortfolioBalanceEstimationResult_State {
    pub estimate: f64,
    pub stock_allocation: f64,
}
impl From<PortfolioBalanceEstimationResult_State> for WirePortfolioBalanceEstimationState {
    fn from(other: PortfolioBalanceEstimationResult_State) -> Self {
        Self {
            estimate: other.estimate.to_float_wire(1),
            allocation: WireStockAllocation {
                stocks_x100: other.stock_allocation.to_float_wire(100),
            },
        }
    }
}

pub struct PortfolioBalanceEstimationResult_StateChange {
    pub start: PortfolioBalanceEstimationResult_State,
    pub end: PortfolioBalanceEstimationResult_State,
}
impl From<PortfolioBalanceEstimationResult_StateChange>
    for WirePortfolioBalanceEstimationStateChange
{
    fn from(other: PortfolioBalanceEstimationResult_StateChange) -> Self {
        Self {
            start: other.start.into(),
            end: other.end.into(),
        }
    }
}

impl From<VTAndBNDData_PercentageChangeFromLastClose> for WireVtAndBndPercentageChange {
    fn from(other: VTAndBNDData_PercentageChangeFromLastClose) -> Self {
        Self {
            vt_x10000: other.vt.to_float_wire(10000),
            bnd_x10000: other.bnd.to_float_wire(10000),
        }
    }
}

pub enum PortfolioBalanceEstimationResult_WithdrawalOrContribution {
    Contribution(f64),
    Withdrawal(f64),
}

impl From<PortfolioBalanceEstimationResult_WithdrawalOrContribution>
    for WirePortfolioBalanceEstimationActionArgsWithdrawalAndContribution
{
    fn from(other: PortfolioBalanceEstimationResult_WithdrawalOrContribution) -> Self {
        Self {
            withdrawal_or_contribution: match other {
                PortfolioBalanceEstimationResult_WithdrawalOrContribution::Contribution(contribution) => Some(
                    wire_portfolio_balance_estimation_action_args_withdrawal_and_contribution::WithdrawalOrContribution::Contribution(
                         contribution.to_float_wire(1),
                    ),
                ),
                PortfolioBalanceEstimationResult_WithdrawalOrContribution::Withdrawal(withdrawal) => Some(
                    wire_portfolio_balance_estimation_action_args_withdrawal_and_contribution::WithdrawalOrContribution::Withdrawal(
                        withdrawal.to_float_wire(1),
                    ),
                ),
            },
        }
    }
}

#[derive(Clone)]
pub struct PortfolioBalanceEstimationResult_PortfolioUpdate {
    pub amount: f64,
    pub exact_timestamp_ms: i64,
}

impl From<PortfolioBalanceEstimationResult_PortfolioUpdate>
    for WirePortfolioBalanceEstimationActionArgsPlanChangePortfolioUpdate
{
    fn from(other: PortfolioBalanceEstimationResult_PortfolioUpdate) -> Self {
        Self {
            amount: other.amount.to_float_wire(1),
            exact_timestamp_ms: other.exact_timestamp_ms,
        }
    }
}

pub enum PortfolioBalanceEstimationResult_ActionArgs {
    MarketClose {
        vt_and_bnd_percentage_change_from_last_close: VTAndBNDData_PercentageChangeFromLastClose,
    },
    WithdrawalOrContribution {
        withdrawal_or_contribution: PortfolioBalanceEstimationResult_WithdrawalOrContribution,
    },
    MonthlyRebalance {
        stock_allocation: f64,
    },
    PlanChange {
        stock_allocation: f64,
        portfolio_update: Option<PortfolioBalanceEstimationResult_PortfolioUpdate>,
    },
}

pub struct PortfolioBalanceEstimationResult_Action {
    pub timestamp_ms: i64,
    pub args: PortfolioBalanceEstimationResult_ActionArgs,
    pub state_change: PortfolioBalanceEstimationResult_StateChange,
}
impl From<PortfolioBalanceEstimationResult_Action> for WirePortfolioBalanceEstimationAction {
    fn from(other: PortfolioBalanceEstimationResult_Action) -> Self {
        Self {
            timestamp_ms: other.timestamp_ms,
            args: match other.args {
                PortfolioBalanceEstimationResult_ActionArgs::MarketClose {
                    vt_and_bnd_percentage_change_from_last_close,
                } => Some(wire_portfolio_balance_estimation_action::Args::MarketClose(
                    WirePortfolioBalanceEstimationActionArgsMarketClose {
                        percentage_change_from_last_close:
                            vt_and_bnd_percentage_change_from_last_close.into(),
                    },
                )),
                PortfolioBalanceEstimationResult_ActionArgs::WithdrawalOrContribution {
                    withdrawal_or_contribution,
                } => Some(
                    wire_portfolio_balance_estimation_action::Args::WithdrawalAndContribution(
                        withdrawal_or_contribution.into(),
                    ),
                ),
                PortfolioBalanceEstimationResult_ActionArgs::MonthlyRebalance {
                    stock_allocation,
                } => Some(
                    wire_portfolio_balance_estimation_action::Args::MonthlyRebalance(
                        WirePortfolioBalanceEstimationActionArgsMonthlyRebalance {
                            allocation: WireStockAllocation {
                                stocks_x100: stock_allocation.to_float_wire(100),
                            },
                        },
                    ),
                ),
                PortfolioBalanceEstimationResult_ActionArgs::PlanChange {
                    stock_allocation,
                    portfolio_update,
                } => Some(wire_portfolio_balance_estimation_action::Args::PlanChange(
                    WirePortfolioBalanceEstimationActionArgsPlanChange {
                        allocation: WireStockAllocation {
                            stocks_x100: stock_allocation.to_float_wire(100),
                        },
                        portfolio_update_opt: portfolio_update.map(|x| x.into()),
                    },
                )),
            },
            state_change: other.state_change.into(),
        }
    }
}
pub struct PortfolioBalanceEstimationResult {
    pub start_timestamp_ms: i64,
    pub end_timestamp_ms: i64,
    pub start_state: PortfolioBalanceEstimationResult_State,
    pub actions: Vec<PortfolioBalanceEstimationResult_Action>,
}

impl From<PortfolioBalanceEstimationResult> for WirePortfolioBalanceEstimationResult {
    fn from(other: PortfolioBalanceEstimationResult) -> Self {
        Self {
            start_timestamp_ms: other.start_timestamp_ms,
            end_timestamp_ms: other.end_timestamp_ms,
            start_state: other.start_state.into(),
            actions: other.actions.into_iter().map(|x| x.into()).collect(),
        }
    }
}
