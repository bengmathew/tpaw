import { PlanParams, TPAWRisk, TPAWRiskLevel } from './PlanParams';
declare type MarketData = {
    CAPE: {
        suggested: number;
        oneOverCAPE: number;
        regressionAverage: number;
    };
    bondRates: {
        twentyYear: number;
    };
    inflation: {
        value: number;
    };
};
export declare const EXPECTED_RETURN_PRESETS: (type: Exclude<PlanParams['returns']['expected']['type'], 'manual'>, { CAPE, bondRates }: MarketData) => {
    stocks: number;
    bonds: number;
};
export declare const SUGGESTED_INFLATION: (marketData: MarketData) => number;
export declare const TPAW_RISK_PRESETS: (preset: Exclude<TPAWRiskLevel, 'custom'>, numYears: number) => TPAWRisk;
export declare const resolveTPAWRiskPreset: (risk: PlanParams['risk'], numYears: number) => Exclude<PlanParams['risk'], {
    useTPAWPreset: true;
}>;
export declare const DEFAULT_SWR_WITHDRAWAL_PERCENT: (retirementLength: number) => number;
export declare function getDefaultPlanParams(): import("./PlanParams14").PlanParams14.Params;
export {};
