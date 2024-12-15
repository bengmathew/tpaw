#![allow(non_camel_case_types)]
pub mod portfolio_balance_estimation_args;
pub mod portfolio_balance_estimation_result;

use std::collections::HashMap;

use itertools::Itertools;
use portfolio_balance_estimation_args::PortfolioBalanceEstimationArgs_ParamsForNonMarketAction;
use portfolio_balance_estimation_result::{
    PortfolioBalanceEstimationResult, PortfolioBalanceEstimationResult_Action,
    PortfolioBalanceEstimationResult_ActionArgs, PortfolioBalanceEstimationResult_PortfolioUpdate,
    PortfolioBalanceEstimationResult_State, PortfolioBalanceEstimationResult_StateChange,
    PortfolioBalanceEstimationResult_WithdrawalOrContribution,
};

use crate::{
    market_data::market_data_defs::{
        MarketDataSeriesForPortfolioBalanceEstimation, MarketDataSeriesForSimulation,
    },
    simulate::{self, plan_params_server::PlanParamsServer_PortfolioBalance},
    wire::WirePortfolioBalanceEstimationArgsNonMarketActionType,
};

struct _ActionSpec<'a> {
    timestamp_ms: i64,
    args_fn: Box<dyn Fn(f64) -> PortfolioBalanceEstimationResult_ActionArgs + 'a>,
}

pub fn estimate_portfolio_balance(
    portfolio_balance_at_history_start: f64,
    mut plan_params_for_actions_unsorted: Vec<
        PortfolioBalanceEstimationArgs_ParamsForNonMarketAction,
    >,
    end_timestamp_ms: i64,
    market_data_series_for_portfolio_balance_estimation: & MarketDataSeriesForPortfolioBalanceEstimation,
    market_data_series_for_simulation: &MarketDataSeriesForSimulation,
) -> PortfolioBalanceEstimationResult {
    plan_params_for_actions_unsorted.sort_by_key(|x| x.plan_params.evaluation_timestamp_ms);
    let plan_params_for_actions = plan_params_for_actions_unsorted;

    let start_timestamp_ms = plan_params_for_actions
        .first()
        .unwrap()
        .plan_params
        .evaluation_timestamp_ms;
    assert!(start_timestamp_ms <= end_timestamp_ms);

    let plan_params_for_actions_by_timestamp_ms: HashMap<
        i64,
        &PortfolioBalanceEstimationArgs_ParamsForNonMarketAction,
    > = plan_params_for_actions
        .iter()
        .map(|x| (x.plan_params.evaluation_timestamp_ms, x))
        .collect();

    struct StockAllocationAndWithdrawal {
        stock_allocation: f64,
        withdrawal_or_contribution: PortfolioBalanceEstimationResult_WithdrawalOrContribution,
    }
    let do_sim = |estimate: f64, timestamp_ms: i64| {
        let history_item = plan_params_for_actions_by_timestamp_ms
            .get(&timestamp_ms)
            .unwrap();

        let (result, plan_params_processed) = simulate::simulate(
            estimate,
            &vec![0_u32],
            1,
            &history_item.plan_params,
            history_item.plan_params.evaluation_timestamp_ms,
            market_data_series_for_simulation,
        );

        let total_withdrawal = result
            .arrays
            .by_percentile_by_mfn_simulated_percentile_major_withdrawals_total[0];
        let contributions = plan_params_processed
            .amount_timed
            .future_savings
            .total_by_mfn[0]
            + plan_params_processed
                .amount_timed
                .income_during_retirement
                .total_by_mfn[0];
        let net_withdrawal = total_withdrawal - contributions;

        StockAllocationAndWithdrawal {
            stock_allocation: result
                .arrays
                .by_percentile_by_mfn_simulated_percentile_major_after_withdrawals_allocation_savings_portfolio[0] as f64,
            withdrawal_or_contribution: if net_withdrawal < 0.0 {
                PortfolioBalanceEstimationResult_WithdrawalOrContribution::Contribution(
                    net_withdrawal.abs(),
                )
            } else {
                PortfolioBalanceEstimationResult_WithdrawalOrContribution::Withdrawal(
                    net_withdrawal,
                )
            },
        }
    };

    // --------------------
    // Market Close Actions
    // --------------------
    let market_close_action_specs = market_data_series_for_portfolio_balance_estimation
        .vt_and_bnd_series
        .iter()
        .filter(|x| {
            // We might make a case for >= starting_timestamp_ms, but to be
            // consistent with server side, we leave it as >
            // starting_timestamp_ms.
            x.closing_time_ms > start_timestamp_ms && x.closing_time_ms <= end_timestamp_ms
        })
        .map(|x| _ActionSpec {
            timestamp_ms: x.closing_time_ms,
            args_fn: Box::new(move |_: f64| {
                PortfolioBalanceEstimationResult_ActionArgs::MarketClose {
                    vt_and_bnd_percentage_change_from_last_close: x
                        .percentage_change_from_last_close
                        .clone(),
                }
            }),
        })
        .collect_vec();

    // -------------------
    // Withdarwal And Contribution Actions
    // -------------------
    let withdrawal_and_contribution_action_specs = plan_params_for_actions
        .iter()
        .filter(|x| {
            x.action_type
                == WirePortfolioBalanceEstimationArgsNonMarketActionType::WithdrawalAndContribution
        })
        .map(|item| _ActionSpec {
            timestamp_ms: item.plan_params.evaluation_timestamp_ms,
            args_fn: Box::new(move |estimate: f64| {
                PortfolioBalanceEstimationResult_ActionArgs::WithdrawalOrContribution {
                    withdrawal_or_contribution: do_sim(
                        estimate,
                        item.plan_params.evaluation_timestamp_ms,
                    )
                    .withdrawal_or_contribution,
                }
            }),
        })
        .collect_vec();

    // -------------------
    // Monthly Rebalance Actions
    // -------------------
    let monthly_rebalance_action_specs = plan_params_for_actions
        .iter()
        .filter(|x| {
            x.action_type == WirePortfolioBalanceEstimationArgsNonMarketActionType::MonthlyRebalance
        })
        .map(|item| _ActionSpec {
            timestamp_ms: item.plan_params.evaluation_timestamp_ms,
            args_fn: Box::new(move |estimate: f64| {
                PortfolioBalanceEstimationResult_ActionArgs::MonthlyRebalance {
                    stock_allocation: do_sim(estimate, item.plan_params.evaluation_timestamp_ms)
                        .stock_allocation,
                }
            }),
        })
        .collect_vec();

    // -------------------
    // Plan Change Actions
    // -------------------
    let plan_change_action_items = plan_params_for_actions
        .iter()
        .filter(|x| {
            x.action_type == WirePortfolioBalanceEstimationArgsNonMarketActionType::PlanChange
        })
        .collect_vec();
    let plan_change_action_specs = plan_change_action_items
        .windows(2)
        .map(|plan_params_window| {
            let prev = &plan_params_window[0];
            let curr = &plan_params_window[1];
            let prev_portfolio_balance_changed_at_id =
                match &prev.plan_params.wealth.portfolio_balance {
                    PlanParamsServer_PortfolioBalance::UpdatedHere(_) => &prev.id,
                    PlanParamsServer_PortfolioBalance::NotUpdatedHere { updated_at_id, .. } => {
                        &updated_at_id
                    }
                };

            let portfolio_update = match &curr.plan_params.wealth.portfolio_balance {
                PlanParamsServer_PortfolioBalance::UpdatedHere(amount) => {
                    Some(PortfolioBalanceEstimationResult_PortfolioUpdate {
                        amount: *amount,
                        exact_timestamp_ms: curr.plan_params.evaluation_timestamp_ms,
                    })
                }
                PlanParamsServer_PortfolioBalance::NotUpdatedHere {
                    updated_at_id,
                    updated_to,
                    updated_at_timestamp_ms,
                } => {
                    if updated_at_id == prev_portfolio_balance_changed_at_id {
                        None
                    } else {
                        Some(PortfolioBalanceEstimationResult_PortfolioUpdate {
                            amount: *updated_to,
                            exact_timestamp_ms: *updated_at_timestamp_ms,
                        })
                    }
                }
            };
            _ActionSpec {
                timestamp_ms: curr.plan_params.evaluation_timestamp_ms,
                args_fn: Box::new(move |estimate: f64| {
                    let effective_estimate = match &portfolio_update {
                        Some(portfolio_update) => portfolio_update.amount,
                        None => estimate,
                    };
                    PortfolioBalanceEstimationResult_ActionArgs::PlanChange {
                        stock_allocation: do_sim(
                            effective_estimate,
                            curr.plan_params.evaluation_timestamp_ms,
                        )
                        .stock_allocation,
                        portfolio_update: portfolio_update.clone(),
                    }
                }),
            }
        })
        .collect_vec();

    let action_specs = _combine_actions_specs(_AllActionSpecs {
        market_close_action_specs,
        monthly_rebalance_action_specs,
        plan_change_action_specs,
        withdrawal_and_contribution_action_specs,
    });

    let start_state = PortfolioBalanceEstimationResult_State {
        estimate: portfolio_balance_at_history_start,
        stock_allocation: do_sim(portfolio_balance_at_history_start, start_timestamp_ms)
            .stock_allocation,
    };

    let mut curr_state = start_state.clone();
    let mut actions: Vec<PortfolioBalanceEstimationResult_Action> = vec![];
    for spec in action_specs {
        let action = _apply_action(spec, &curr_state);
        curr_state = action.state_change.end.clone();
        actions.push(action);
    }
    PortfolioBalanceEstimationResult {
        start_timestamp_ms,
        end_timestamp_ms,
        start_state,
        actions,
    }
}

// Note: This is mirrored from packages/common. See there for more details.
struct _AllActionSpecs<'a> {
    market_close_action_specs: Vec<_ActionSpec<'a>>,
    monthly_rebalance_action_specs: Vec<_ActionSpec<'a>>,
    plan_change_action_specs: Vec<_ActionSpec<'a>>,
    withdrawal_and_contribution_action_specs: Vec<_ActionSpec<'a>>,
}
fn _combine_actions_specs<'a>(mut x: _AllActionSpecs<'a>) -> Vec<_ActionSpec<'a>> {
    let mut combined: Vec<_ActionSpec<'a>> = vec![];
    combined.append(&mut x.market_close_action_specs);
    combined.append(&mut x.monthly_rebalance_action_specs);
    combined.append(&mut x.plan_change_action_specs);
    combined.append(&mut x.withdrawal_and_contribution_action_specs);
    // This is stable sort.
    combined.sort_by_key(|x| x.timestamp_ms);
    combined
}

fn _apply_action(
    spec: _ActionSpec,
    start: &PortfolioBalanceEstimationResult_State,
) -> PortfolioBalanceEstimationResult_Action {
    let args = (spec.args_fn)(start.estimate);
    let end = match &args {
        PortfolioBalanceEstimationResult_ActionArgs::MarketClose {
            vt_and_bnd_percentage_change_from_last_close,
        } => {
            let result = PortfolioBalanceEstimationResult_State {
                estimate: (1.0 + vt_and_bnd_percentage_change_from_last_close.vt)
                    * start.estimate
                    * start.stock_allocation
                    + (1.0 + vt_and_bnd_percentage_change_from_last_close.bnd)
                        * start.estimate
                        * (1.0 - start.stock_allocation),
                stock_allocation: start.stock_allocation,
            };
            result
        }
        PortfolioBalanceEstimationResult_ActionArgs::WithdrawalOrContribution {
            withdrawal_or_contribution,
        } => PortfolioBalanceEstimationResult_State {
            estimate: start.estimate
                + match withdrawal_or_contribution {
                    PortfolioBalanceEstimationResult_WithdrawalOrContribution::Contribution(
                        contribution,
                    ) => *contribution,
                    PortfolioBalanceEstimationResult_WithdrawalOrContribution::Withdrawal(
                        withdrawal,
                    ) => -*withdrawal,
                },
            stock_allocation: start.stock_allocation,
        },
        PortfolioBalanceEstimationResult_ActionArgs::MonthlyRebalance { stock_allocation } => {
            PortfolioBalanceEstimationResult_State {
                estimate: start.estimate,
                stock_allocation: *stock_allocation,
            }
        }
        PortfolioBalanceEstimationResult_ActionArgs::PlanChange {
            stock_allocation,
            portfolio_update,
        } => PortfolioBalanceEstimationResult_State {
            estimate: match portfolio_update {
                Some(portfolio_update) => portfolio_update.amount,
                None => start.estimate,
            },
            stock_allocation: *stock_allocation,
        },
    };
    PortfolioBalanceEstimationResult_Action {
        timestamp_ms: spec.timestamp_ms,
        args,
        state_change: PortfolioBalanceEstimationResult_StateChange {
            start: start.clone(),
            end,
        },
    }
}
