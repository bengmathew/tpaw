"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.PlanParams14 = void 0;
// import {getDefaultParams} from './DefaultPlanParams'
// import {TPAWParamsV13} from './TPAWParamsOld/TPAWParamsV13'
const json_guard_1 = require("json-guard");
const lodash_1 = __importDefault(require("lodash"));
const Utils_1 = require("../Utils");
const DefaultPlanParams_1 = require("./DefaultPlanParams");
const TPAWParamsV10_1 = require("./Old/TPAWParamsV10");
const TPAWParamsV11_1 = require("./Old/TPAWParamsV11");
const TPAWParamsV12_1 = require("./Old/TPAWParamsV12");
const TPAWParamsV13_1 = require("./Old/TPAWParamsV13");
const TPAWParamsV1Validator_1 = require("./Old/TPAWParamsV1Validator");
const TPAWParamsV2Validator_1 = require("./Old/TPAWParamsV2Validator");
const TPAWParamsV3Validator_1 = require("./Old/TPAWParamsV3Validator");
const TPAWParamsV4_1 = require("./Old/TPAWParamsV4");
const TPAWParamsV5_1 = require("./Old/TPAWParamsV5");
const TPAWParamsV6_1 = require("./Old/TPAWParamsV6");
const TPAWParamsV7_1 = require("./Old/TPAWParamsV7");
const TPAWParamsV8_1 = require("./Old/TPAWParamsV8");
const TPAWParamsV9_1 = require("./Old/TPAWParamsV9");
const Validator_1 = require("./Old/Validator");
var PlanParams14;
(function (PlanParams14) {
    PlanParams14.MAX_LABEL_LENGTH = 150;
    PlanParams14.MAX_AGE = 120;
    PlanParams14.MAX_NUM_YEARS_IN_GLIDE_PATH = 1000;
    PlanParams14.MAX_VALUE_FOR_YEAR_RANGE = 100;
    PlanParams14.MAX_EXTERNAL_LEGACY_SOURCES = 100;
    // ----------- VALIDATOR  ---------//
    const _ageRange = (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0), (0, json_guard_1.lte)(PlanParams14.MAX_AGE));
    const _ages = (0, json_guard_1.chain)((0, json_guard_1.union)((0, json_guard_1.object)({
        type: (0, json_guard_1.constant)('retired'),
        current: _ageRange,
        max: _ageRange,
    }), (0, json_guard_1.object)({
        type: (0, json_guard_1.constant)('notRetired'),
        current: _ageRange,
        retirement: _ageRange,
        max: _ageRange,
    })), (ages) => {
        const { current, max } = ages;
        if (max < current + 1) {
            return (0, json_guard_1.failure)('Max age should be at least one year after current age.');
        }
        if (ages.type === 'notRetired') {
            const { retirement } = ages;
            if (retirement < current + 1) {
                return (0, json_guard_1.failure)('Retirement age should be at least one year after current age.');
            }
            if (max < retirement + 1) {
                return (0, json_guard_1.failure)('Max age should be at least one year after retirement age.');
            }
        }
        return (0, json_guard_1.success)(ages);
    });
    const person = (0, json_guard_1.object)({
        ages: _ages,
        displayName: (0, json_guard_1.nullable)((0, json_guard_1.chain)(json_guard_1.string, (0, json_guard_1.bounded)(PlanParams14.MAX_LABEL_LENGTH))),
    });
    const people = (0, json_guard_1.union)((0, json_guard_1.object)({
        withPartner: (0, json_guard_1.constant)(false),
        person1: person,
    }), (0, json_guard_1.object)({
        withPartner: (0, json_guard_1.constant)(true),
        person2: person,
        person1: person,
        withdrawalStart: (0, json_guard_1.union)((0, json_guard_1.constant)('person1'), (0, json_guard_1.constant)('person2')),
        xAxis: (0, json_guard_1.union)((0, json_guard_1.constant)('person1'), (0, json_guard_1.constant)('person2')),
    }));
    const year = (people) => (0, json_guard_1.union)((0, json_guard_1.object)({ type: (0, json_guard_1.constant)('now') }), (0, json_guard_1.chain)((0, json_guard_1.object)({
        type: (0, json_guard_1.constant)('namedAge'),
        person: (0, json_guard_1.union)((0, json_guard_1.constant)('person1'), (0, json_guard_1.constant)('person2')),
        age: (0, json_guard_1.union)((0, json_guard_1.constant)('lastWorkingYear'), (0, json_guard_1.constant)('retirement'), (0, json_guard_1.constant)('max')),
    }), (x) => {
        if (!people)
            return (0, json_guard_1.success)(x);
        let person;
        if (x.person === 'person1') {
            person = people.person1;
        }
        else {
            if (!people.withPartner)
                return (0, json_guard_1.failure)('In terms of partner, but there is no partner.');
            person = people.person2;
        }
        if ((x.age === 'retirement' || x.age === 'lastWorkingYear') &&
            person.ages.type === 'retired') {
            return (0, json_guard_1.failure)(`In terms retirement age of ${x.person}, but ${x.person} is already retired.`);
        }
        return (0, json_guard_1.success)(x);
    }), (0, json_guard_1.chain)((0, json_guard_1.object)({
        type: (0, json_guard_1.constant)('numericAge'),
        person: (0, json_guard_1.union)((0, json_guard_1.constant)('person1'), (0, json_guard_1.constant)('person2')),
        age: (0, json_guard_1.chain)(json_guard_1.number, json_guard_1.integer),
    }), (x) => {
        if (!people)
            return (0, json_guard_1.success)(x);
        if (x.person === 'person2' && !people.withPartner)
            return (0, json_guard_1.failure)('In terms of partner, but there is no partner.');
        return (0, json_guard_1.success)(x);
    }));
    const yearRange = (people) => {
        const yr = year(people);
        return (0, json_guard_1.union)((0, json_guard_1.object)({ type: (0, json_guard_1.constant)('startAndEnd'), start: yr, end: yr }), (0, json_guard_1.object)({
            type: (0, json_guard_1.constant)('startAndNumYears'),
            start: yr,
            numYears: (0, json_guard_1.chain)(json_guard_1.number, json_guard_1.integer, (0, json_guard_1.gt)(0)),
        }), (0, json_guard_1.object)({
            type: (0, json_guard_1.constant)('endAndNumYears'),
            end: yr,
            numYears: (0, json_guard_1.chain)(json_guard_1.number, json_guard_1.integer, (0, json_guard_1.gt)(0)),
        }));
    };
    const valueForYearRange = (people) => (0, json_guard_1.object)({
        // Not trimmed because it won't allow space even temporarily.
        label: (0, json_guard_1.nullable)((0, json_guard_1.chain)(json_guard_1.string, (0, json_guard_1.bounded)(PlanParams14.MAX_LABEL_LENGTH))),
        yearRange: yearRange(people),
        value: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0)),
        nominal: json_guard_1.boolean,
        id: (0, json_guard_1.chain)(json_guard_1.number, json_guard_1.integer, (0, json_guard_1.gte)(0)),
    });
    const valueForYearRangeArr = (people = null) => (0, json_guard_1.array)(valueForYearRange(people), PlanParams14.MAX_VALUE_FOR_YEAR_RANGE);
    const extraSpending = (people = null) => (0, json_guard_1.object)({
        essential: (0, json_guard_1.array)(valueForYearRange(people), PlanParams14.MAX_VALUE_FOR_YEAR_RANGE),
        discretionary: (0, json_guard_1.array)(valueForYearRange(people), PlanParams14.MAX_VALUE_FOR_YEAR_RANGE),
    });
    const glidePath = (params) => {
        var _a;
        return (0, json_guard_1.object)({
            start: (0, json_guard_1.object)({ stocks: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0), (0, json_guard_1.lte)(1)) }),
            intermediate: (0, json_guard_1.array)((0, json_guard_1.object)({
                year: year((_a = params === null || params === void 0 ? void 0 : params.people) !== null && _a !== void 0 ? _a : null),
                stocks: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0), (0, json_guard_1.lte)(1)),
            }), PlanParams14.MAX_NUM_YEARS_IN_GLIDE_PATH - 2),
            end: (0, json_guard_1.object)({ stocks: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0), (0, json_guard_1.lte)(1)) }),
        });
    };
    const tpawRisk = (params) => (0, json_guard_1.object)({
        tpaw: (0, json_guard_1.object)({
            allocation: glidePath(params),
            allocationForLegacy: (0, json_guard_1.object)({
                stocks: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0), (0, json_guard_1.lte)(1)),
            }),
        }),
        tpawAndSPAW: (0, json_guard_1.chain)((0, json_guard_1.object)({
            spendingCeiling: (0, json_guard_1.nullable)((0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0))),
            spendingFloor: (0, json_guard_1.nullable)((0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0))),
            spendingTilt: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(-0.03), (0, json_guard_1.lte)(0.03)),
            lmp: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0)),
        }), (x) => {
            var _a;
            return x.spendingCeiling !== null &&
                ((_a = x.spendingFloor) !== null && _a !== void 0 ? _a : 0) > x.spendingCeiling
                ? (0, json_guard_1.failure)('Spending Floor is greater than spending ceiling.')
                : (0, json_guard_1.success)(x);
        }),
    }, 'extraKeysOk');
    const risk = (params) => (0, json_guard_1.intersection)((0, json_guard_1.union)((0, json_guard_1.object)({
        useTPAWPreset: (0, json_guard_1.chain)((0, json_guard_1.constant)(true), (x) => !params || params.strategy === 'TPAW'
            ? (0, json_guard_1.success)(x)
            : (0, json_guard_1.failure)('useTPAWPreset is true but strategy is not TPAW.')),
    }, 'extraKeysOk'), (0, json_guard_1.intersection)((0, json_guard_1.object)({ useTPAWPreset: (0, json_guard_1.constant)(false) }, 'extraKeysOk'), tpawRisk(params))), (0, json_guard_1.object)({
        tpawPreset: (0, json_guard_1.union)((0, json_guard_1.constant)('riskLevel-1'), (0, json_guard_1.constant)('riskLevel-2'), (0, json_guard_1.constant)('riskLevel-3'), (0, json_guard_1.constant)('riskLevel-4')),
        customTPAWPreset: (0, json_guard_1.nullable)(tpawRisk(params)),
        savedTPAWPreset: (0, json_guard_1.nullable)(tpawRisk(params)),
        swr: (0, json_guard_1.object)({
            withdrawal: (0, json_guard_1.union)((0, json_guard_1.object)({
                type: (0, json_guard_1.constant)('asPercent'),
                percent: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0), (0, json_guard_1.lte)(1)),
            }), (0, json_guard_1.object)({
                type: (0, json_guard_1.constant)('asAmount'),
                amount: (0, json_guard_1.chain)(json_guard_1.number, json_guard_1.integer, (0, json_guard_1.gte)(0)),
            }), (0, json_guard_1.object)({ type: (0, json_guard_1.constant)('default') })),
        }),
        spawAndSWR: (0, json_guard_1.object)({
            allocation: glidePath(params),
        }),
    }, 'extraKeysOk'));
    const returns = (0, json_guard_1.object)({
        expected: (0, json_guard_1.union)((0, json_guard_1.object)({ type: (0, json_guard_1.constant)('suggested') }), (0, json_guard_1.object)({ type: (0, json_guard_1.constant)('oneOverCAPE') }), (0, json_guard_1.object)({ type: (0, json_guard_1.constant)('regressionPrediction') }), (0, json_guard_1.object)({ type: (0, json_guard_1.constant)('historical') }), (0, json_guard_1.object)({
            type: (0, json_guard_1.constant)('manual'),
            stocks: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(-0.01), (0, json_guard_1.lte)(0.1)),
            bonds: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(-0.01), (0, json_guard_1.lte)(0.1)),
        })),
        historical: (0, json_guard_1.union)((0, json_guard_1.object)({
            type: (0, json_guard_1.constant)('default'),
            adjust: (0, json_guard_1.union)((0, json_guard_1.object)({
                type: (0, json_guard_1.constant)('to'),
                stocks: json_guard_1.number,
                bonds: json_guard_1.number,
            }), (0, json_guard_1.object)({
                type: (0, json_guard_1.constant)('by'),
                stocks: json_guard_1.number,
                bonds: json_guard_1.number,
            }), (0, json_guard_1.object)({ type: (0, json_guard_1.constant)('toExpected') }), (0, json_guard_1.object)({ type: (0, json_guard_1.constant)('none') })),
        }), (0, json_guard_1.object)({
            type: (0, json_guard_1.constant)('fixed'),
            stocks: json_guard_1.number,
            bonds: json_guard_1.number,
        })),
    });
    const legacy = (0, json_guard_1.object)({
        tpawAndSPAW: (0, json_guard_1.object)({
            total: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0)),
            external: (0, json_guard_1.array)((0, json_guard_1.object)({
                label: (0, json_guard_1.nullable)((0, json_guard_1.chain)(json_guard_1.string, (0, json_guard_1.bounded)(PlanParams14.MAX_LABEL_LENGTH))),
                value: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0)),
                nominal: json_guard_1.boolean,
            }), PlanParams14.MAX_EXTERNAL_LEGACY_SOURCES),
        }),
    });
    const inflation = (0, json_guard_1.union)((0, json_guard_1.object)({ type: (0, json_guard_1.constant)('suggested') }), (0, json_guard_1.object)({
        type: (0, json_guard_1.constant)('manual'),
        value: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(-0.01), (0, json_guard_1.lte)(0.1)),
    }));
    const params = (x) => {
        var _a, _b, _c;
        return (0, json_guard_1.object)({
            v: (0, json_guard_1.constant)(14),
            strategy: (0, json_guard_1.union)((0, json_guard_1.constant)('TPAW'), (0, json_guard_1.constant)('SPAW'), (0, json_guard_1.constant)('SWR')),
            dialogMode: json_guard_1.boolean,
            people,
            currentPortfolioBalance: (0, json_guard_1.chain)(json_guard_1.number, (0, json_guard_1.gte)(0)),
            futureSavings: valueForYearRangeArr((_a = x === null || x === void 0 ? void 0 : x.people) !== null && _a !== void 0 ? _a : null),
            retirementIncome: valueForYearRangeArr((_b = x === null || x === void 0 ? void 0 : x.people) !== null && _b !== void 0 ? _b : null),
            extraSpending: extraSpending((_c = x === null || x === void 0 ? void 0 : x.people) !== null && _c !== void 0 ? _c : null),
            legacy,
            risk: risk(x),
            returns,
            inflation,
            sampling: (0, json_guard_1.union)((0, json_guard_1.constant)('monteCarlo'), (0, json_guard_1.constant)('historical')),
            display: (0, json_guard_1.object)({ alwaysShowAllYears: json_guard_1.boolean }),
        });
    };
    const v14Guard = (0, json_guard_1.chain)(params(null), (x) => params(x)(x));
    PlanParams14.guard = (x) => {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
        if (x.v === 14) {
            return v14Guard(x);
        }
        else {
            const v13Result = v13Guard(x);
            if (v13Result.error)
                return v13Result;
            const v13 = v13Result.value;
            const v14 = {
                ...v13,
                v: 14,
                risk: v13.risk.tpawPreset === 'custom'
                    ? {
                        useTPAWPreset: false,
                        tpawPreset: (0, DefaultPlanParams_1.getDefaultPlanParams)().risk.tpawPreset,
                        customTPAWPreset: v13.risk.customTPAWPreset,
                        savedTPAWPreset: null,
                        tpaw: lodash_1.default.cloneDeep((0, Utils_1.fGet)(v13.risk.customTPAWPreset).tpaw),
                        tpawAndSPAW: lodash_1.default.cloneDeep((0, Utils_1.fGet)(v13.risk.customTPAWPreset).tpawAndSPAW),
                        swr: v13.risk.swr,
                        spawAndSWR: v13.risk.spawAndSWR,
                    }
                    : {
                        ...v13.risk,
                        tpawPreset: v13.risk.tpawPreset,
                        savedTPAWPreset: null,
                    },
            };
            return (0, json_guard_1.success)(v14Guard(v14).force());
        }
    };
})(PlanParams14 = exports.PlanParams14 || (exports.PlanParams14 = {}));
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const v13Guard = (parsed) => {
    try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
        const version = 'v' in parsed ? parsed.v : 1;
        if (typeof version !== 'number')
            return (0, json_guard_1.failure)('Version is not a number.');
        if (version > 13 || version < 0)
            return (0, json_guard_1.failure)('Invalid version number.');
        const v1 = version === 1 ? (0, TPAWParamsV1Validator_1.tpawParamsV1Validator)(parsed) : null;
        const v2 = version === 2 ? (0, TPAWParamsV2Validator_1.tpawParamsV2Validator)(parsed) : v1 ? _v1ToV2(v1) : null;
        const v3 = version === 3 ? (0, TPAWParamsV3Validator_1.tpawParamsV3Validator)(parsed) : v2 ? _v2ToV3(v2) : null;
        const v4 = version === 4 ? TPAWParamsV4_1.TPAWParamsV4.validator(parsed) : v3 ? _v3ToV4(v3) : null;
        const v5 = version === 5 ? TPAWParamsV5_1.TPAWParamsV5.validator(parsed) : v4 ? _v4ToV5(v4) : null;
        const v6 = version === 6
            ? TPAWParamsV6_1.TPAWParamsV6.validator(parsed)
            : v5
                ? TPAWParamsV6_1.TPAWParamsV6.fromV5(v5)
                : null;
        const v7 = version === 7
            ? TPAWParamsV7_1.TPAWParamsV7.validator(parsed)
            : v6
                ? TPAWParamsV7_1.TPAWParamsV7.fromV6(v6)
                : null;
        const v8 = version === 8
            ? TPAWParamsV8_1.TPAWParamsV8.validator(parsed)
            : v7
                ? TPAWParamsV8_1.TPAWParamsV8.fromV7(v7)
                : null;
        const v9 = version === 9
            ? TPAWParamsV9_1.TPAWParamsV9.validator(parsed)
            : v8
                ? TPAWParamsV9_1.TPAWParamsV9.fromV8(v8)
                : null;
        const v10 = version === 10
            ? TPAWParamsV10_1.TPAWParamsV10.validator(parsed)
            : v9
                ? TPAWParamsV10_1.TPAWParamsV10.fromV9(v9)
                : null;
        const v11 = version === 11
            ? TPAWParamsV11_1.TPAWParamsV11.validator(parsed)
            : v10
                ? TPAWParamsV11_1.TPAWParamsV11.fromV10(v10)
                : null;
        const v12 = version === 12
            ? TPAWParamsV12_1.TPAWParamsV12.validator(parsed)
            : v11
                ? TPAWParamsV12_1.TPAWParamsV12.fromV11(v11)
                : null;
        return (0, json_guard_1.success)(version === 13
            ? TPAWParamsV13_1.TPAWParamsV13.validator(parsed)
            : TPAWParamsV13_1.TPAWParamsV13.fromV12((0, Utils_1.fGet)(v12)));
    }
    catch (e) {
        if (e instanceof Validator_1.Validator.Failed) {
            return (0, json_guard_1.failure)(e.fullMessage);
        }
        else {
            throw e;
        }
    }
};
const _v1ToV2 = (v1) => {
    const savings = [];
    const retirementIncome = [];
    v1.savings.forEach((x) => {
        const start = _numericYear(v1, x.yearRange.start);
        const end = _numericYear(v1, x.yearRange.end);
        if (start < v1.age.retirement && end >= v1.age.retirement) {
            savings.push({
                ...x,
                yearRange: { ...x.yearRange, end: 'lastWorkingYear' },
            });
            retirementIncome.push({
                ...x,
                yearRange: { ...x.yearRange, start: 'retirement' },
            });
        }
        else {
            start < v1.age.retirement ? savings.push(x) : retirementIncome.push(x);
        }
    });
    return {
        v: 2,
        ...v1,
        savings,
        retirementIncome,
    };
};
const _v2ToV3 = (v2) => {
    return {
        ...lodash_1.default.cloneDeep(v2),
        v: 3,
        spendingFloor: null,
    };
};
const _v3ToV4 = (v3) => {
    const { retirementIncome, withdrawals, ...rest } = lodash_1.default.cloneDeep(v3);
    const addId = (x, id) => ({ ...x, id });
    retirementIncome;
    return {
        ...rest,
        v: 4,
        retirementIncome: retirementIncome.map(addId),
        savings: retirementIncome.map(addId),
        withdrawals: {
            fundedByBonds: withdrawals.fundedByBonds.map(addId),
            fundedByRiskPortfolio: withdrawals.fundedByRiskPortfolio.map(addId),
        },
    };
};
const _v4ToV5 = (v4) => {
    const { age, savings, retirementIncome, withdrawals, ...rest } = v4;
    const year = (year) => year === 'start'
        ? { type: 'now' }
        : typeof year === 'number'
            ? { type: 'numericAge', person: 'person1', age: year }
            : {
                type: 'namedAge',
                person: 'person1',
                age: year === 'end' ? 'max' : year,
            };
    const valueForYearRange = ({ yearRange, ...rest }) => ({
        yearRange: {
            type: 'startAndEnd',
            start: year(yearRange.start),
            end: year(yearRange.end),
        },
        ...rest,
    });
    const result = {
        ...rest,
        v: 5,
        people: {
            withPartner: false,
            person1: {
                ages: age.start === age.retirement
                    ? {
                        type: 'retired',
                        current: age.start,
                        max: age.end,
                    }
                    : {
                        type: 'notRetired',
                        current: age.start,
                        retirement: age.retirement,
                        max: age.end,
                    },
                displayName: null,
            },
        },
        savings: savings.map(valueForYearRange),
        retirementIncome: retirementIncome.map(valueForYearRange),
        withdrawals: {
            fundedByBonds: withdrawals.fundedByBonds.flatMap(valueForYearRange),
            fundedByRiskPortfolio: withdrawals.fundedByRiskPortfolio.flatMap(valueForYearRange),
        },
    };
    TPAWParamsV5_1.TPAWParamsV5.validator(result);
    return result;
};
const _numericYear = ({ age }, x) => x === 'start'
    ? age.start
    : x === 'lastWorkingYear'
        ? age.retirement - 1
        : x === 'retirement'
            ? age.retirement
            : x === 'end'
                ? age.end
                : x;
//# sourceMappingURL=PlanParams14.js.map