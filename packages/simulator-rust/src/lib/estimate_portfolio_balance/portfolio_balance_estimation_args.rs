#![allow(non_camel_case_types)]

use std::convert::TryFrom;

use crate::{
    simulate::plan_params_server::PlanParamsServer,
    wire::{
        WirePortfolioBalanceEstimationArgs, WirePortfolioBalanceEstimationArgsNonMarketActionType,
        WirePortfolioBalanceEstimationArgsParamsForNonMarketAction
    },
};

#[derive(Clone)]
pub struct PortfolioBalanceEstimationArgs_ParamsForNonMarketAction {
    pub id: String,
    pub plan_params: PlanParamsServer,
    pub action_type: WirePortfolioBalanceEstimationArgsNonMarketActionType,
}

impl From<WirePortfolioBalanceEstimationArgsParamsForNonMarketAction>
    for PortfolioBalanceEstimationArgs_ParamsForNonMarketAction
{
    fn from(other: WirePortfolioBalanceEstimationArgsParamsForNonMarketAction) -> Self {
        Self {
            id: other.id,
            plan_params: other.plan_params.into(),
            action_type: WirePortfolioBalanceEstimationArgsNonMarketActionType::try_from(
                other.action_type,
            )
            .unwrap(),
        }
    }
}

pub struct PortfolioBalanceEstimationArgs {
    pub start_balance: f64,
    pub plan_params_for_non_market_actions_unsorted:
        Vec<PortfolioBalanceEstimationArgs_ParamsForNonMarketAction>,
}
impl From<WirePortfolioBalanceEstimationArgs>
    for PortfolioBalanceEstimationArgs
{
    fn from(wire: WirePortfolioBalanceEstimationArgs) -> Self {
        Self {
            start_balance: wire.start_balance,
            plan_params_for_non_market_actions_unsorted: wire
                .plan_params_for_non_market_actions_unsorted
                .into_iter()
                .map(|p| p.into())
                .collect(),
        }
    }
}
