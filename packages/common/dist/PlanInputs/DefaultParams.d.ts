import { TPAWParams, TPAWRiskLevel } from './TPAWParams';
export declare const EXPECTED_RETURN_PRESETS: (type: Exclude<TPAWParams['returns']['expected']['type'], 'manual'>, { CAPE, bondRates }: MarketData) => any;
export declare const SUGGESTED_INFLATION: (marketData: MarketData) => number;
export declare const TPAW_RISK_PRESETS: (preset: Exclude<TPAWRiskLevel, 'custom'>, numYears: number) => TPAWRisk;
export declare const resolveTPAWRiskPreset: (risk: TPAWParams, numYears: number) => Exclude<TPAWParams['risk'], {
    useTPAWPreset: true;
}>;
export declare const DEFAULT_SWR_WITHDRAWAL_PERCENT: (retirementLength: number) => number;
export declare function getDefaultParams(): TPAWParams;
