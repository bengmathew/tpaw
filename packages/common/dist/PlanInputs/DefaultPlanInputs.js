"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getDefaultParams = exports.DEFAULT_SWR_WITHDRAWAL_PERCENT = exports.resolveTPAWRiskPreset = exports.TPAW_RISK_PRESETS = exports.SUGGESTED_INFLATION = exports.EXPECTED_RETURN_PRESETS = void 0;
const lodash_1 = __importDefault(require("lodash"));
const LinearFn_1 = require("../Utils/LinearFn");
const Utils_1 = require("../Utils/Utils");
const HistoricalReturns_1 = require("./HistoricalReturns");
const EXPECTED_RETURN_PRESETS = (type, { CAPE, bondRates }) => {
    const suggested = {
        stocks: lodash_1.default.round(CAPE.suggested, 3),
        bonds: lodash_1.default.round(bondRates.twentyYear, 3),
    };
    switch (type) {
        case 'suggested':
            return { ...suggested };
        case 'oneOverCAPE':
            return {
                stocks: lodash_1.default.round(CAPE.oneOverCAPE, 3),
                bonds: suggested.bonds,
            };
        case 'regressionPrediction':
            return {
                stocks: lodash_1.default.round(CAPE.regressionAverage, 3),
                bonds: suggested.bonds,
            };
        case 'historical':
            return { ...HistoricalReturns_1.historicalReturnsAverage };
        default:
            (0, Utils_1.noCase)(type);
    }
};
exports.EXPECTED_RETURN_PRESETS = EXPECTED_RETURN_PRESETS;
const SUGGESTED_INFLATION = (marketData) => lodash_1.default.round(marketData.inflation.value, 3);
exports.SUGGESTED_INFLATION = SUGGESTED_INFLATION;
const TPAW_RISK_PRESETS = (preset, numYears) => {
    const allocFn = (at20of100, spendingTilt) => {
        const at100of100 = lodash_1.default.round(at20of100 - 0.1, 2);
        return {
            tpaw: {
                allocation: {
                    start: {
                        stocks: lodash_1.default.round((0, LinearFn_1.linearFnFomPoints)(0, at100of100, 100 - 20, at20of100)(numYears), 2),
                    },
                    intermediate: [],
                    end: { stocks: at100of100 },
                },
                allocationForLegacy: { stocks: lodash_1.default.round(at20of100 + 0.2, 2) },
            },
            tpawAndSPAW: {
                spendingCeiling: null,
                spendingFloor: null,
                spendingTilt,
                lmp: 0,
            },
        };
    };
    switch (preset) {
        case 'riskLevel-1':
            return allocFn(0.3, 0.005);
        case 'riskLevel-2':
            return allocFn(0.4, 0.008);
        case 'riskLevel-3':
            return allocFn(0.5, 0.011);
        case 'riskLevel-4':
            return allocFn(0.6, 0.014);
        default:
            (0, Utils_1.noCase)(preset);
    }
};
exports.TPAW_RISK_PRESETS = TPAW_RISK_PRESETS;
const resolveTPAWRiskPreset = (risk, numYears) => {
    if (!risk.useTPAWPreset)
        return risk;
    const { swr, spawAndSWR, tpawPreset, customTPAWPreset, savedTPAWPreset } = risk;
    const { tpaw, tpawAndSPAW } = (0, exports.TPAW_RISK_PRESETS)(risk.tpawPreset, numYears);
    return {
        useTPAWPreset: false,
        tpaw,
        tpawAndSPAW,
        swr,
        spawAndSWR,
        tpawPreset,
        customTPAWPreset,
        savedTPAWPreset,
    };
};
exports.resolveTPAWRiskPreset = resolveTPAWRiskPreset;
const DEFAULT_SWR_WITHDRAWAL_PERCENT = (retirementLength) => {
    return lodash_1.default.round(0.7125 * Math.pow(retirementLength, -0.859), 3);
};
exports.DEFAULT_SWR_WITHDRAWAL_PERCENT = DEFAULT_SWR_WITHDRAWAL_PERCENT;
function getDefaultParams() {
    const params = {
        v: 14,
        strategy: 'TPAW',
        dialogMode: true,
        people: {
            withPartner: false,
            person1: {
                displayName: null,
                ages: { type: 'notRetired', current: 35, retirement: 65, max: 100 },
            },
        },
        currentPortfolioBalance: 0,
        futureSavings: [],
        retirementIncome: [],
        extraSpending: {
            essential: [],
            discretionary: [],
        },
        legacy: {
            tpawAndSPAW: {
                total: 0,
                external: [],
            },
        },
        risk: {
            useTPAWPreset: true,
            tpawPreset: 'riskLevel-2',
            customTPAWPreset: null,
            savedTPAWPreset: null,
            spawAndSWR: {
                allocation: {
                    start: { stocks: 0.5 },
                    intermediate: [],
                    end: { stocks: 0.5 },
                },
            },
            swr: {
                withdrawal: { type: 'default' },
            },
        },
        returns: {
            expected: { type: 'suggested' },
            historical: { type: 'default', adjust: { type: 'toExpected' } },
        },
        inflation: { type: 'suggested' },
        sampling: 'monteCarlo',
        display: {
            alwaysShowAllYears: false,
        },
    };
    return params;
}
exports.getDefaultParams = getDefaultParams;
//# sourceMappingURL=DefaultPlanInputs.js.map