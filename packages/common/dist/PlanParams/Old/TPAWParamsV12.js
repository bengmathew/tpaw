"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.TPAWParamsV12 = void 0;
const lodash_1 = __importDefault(require("lodash"));
const Validator_1 = require("./Validator");
var TPAWParamsV12;
(function (TPAWParamsV12) {
    TPAWParamsV12.MAX_LABEL_LENGTH = 150;
    TPAWParamsV12.MAX_AGE = 120;
    TPAWParamsV12.MAX_NUM_YEARS_IN_GLIDE_PATH = 1000;
    const { number, string, constant, chain, object, union, array, boolean } = Validator_1.Validator;
    const numRangeInclusive = (min, max) => chain(number(), _geTest(min), _leTest(max));
    const numGE = (min) => chain(number(), _geTest(min));
    const numInt = () => chain(number(), _intTest);
    const numIntNonNeg = () => chain(numInt(), _geTest(0));
    const numIntPos = () => chain(numInt(), _geTest(1));
    const strBounded = (maxLength) => chain(string(), _boundedStringTest(maxLength));
    const strBoundedTrimmed = (maxLength) => chain(strBounded(maxLength), _trimmedTest);
    const sizedArray = (test, min, max) => chain(array(test), x => {
        if (x.length < min)
            throw new Validator_1.Validator.Failed(`Less than ${min} elements.`);
        if (x.length > max)
            throw new Validator_1.Validator.Failed(`Greater than ${max} elements.`);
        return x;
    });
    const _intTest = (x) => {
        if (!lodash_1.default.isInteger(x))
            throw new Validator_1.Validator.Failed('Not an integer.');
        return x;
    };
    const _boundedStringTest = (bound) => (x) => {
        if (x.length > bound)
            throw new Validator_1.Validator.Failed('Too large.');
        return x;
    };
    const _trimmedTest = (x) => {
        if (x.trim().length !== x.length)
            throw new Validator_1.Validator.Failed('Not trimmed.');
        return x;
    };
    const _geTest = (y) => (x) => {
        if (x < y)
            throw new Validator_1.Validator.Failed(`Less than ${y}.`);
        return x;
    };
    const _leTest = (y) => (x) => {
        if (x > y)
            throw new Validator_1.Validator.Failed(`Greater than ${y}.`);
        return x;
    };
    // ---------- FROM V10 ------------//
    TPAWParamsV12.fromV11 = (params) => {
        const result = {
            ...params,
            targetAllocation: {
                ...params.targetAllocation,
                regularPortfolio: {
                    ...params.targetAllocation.regularPortfolio,
                    forTPAW: {
                        start: { ...params.targetAllocation.regularPortfolio.forTPAW },
                        intermediate: [],
                        end: { ...params.targetAllocation.regularPortfolio.forTPAW },
                    },
                },
            },
            v: 12,
        };
        TPAWParamsV12.validator(result);
        return result;
    };
    // ----------- VALIDATOR  ---------//
    const _ageRange = chain(number(), _geTest(0), _leTest(TPAWParamsV12.MAX_AGE));
    const _ages = chain(union(object({
        type: constant('retired'),
        current: _ageRange,
        max: _ageRange,
    }), object({
        type: constant('notRetired'),
        current: _ageRange,
        retirement: _ageRange,
        max: _ageRange,
    })), (ages) => {
        const { current, max } = ages;
        if (max < current + 1) {
            throw new Validator_1.Validator.Failed('Max age should be at least one year after current age.');
        }
        if (ages.type === 'notRetired') {
            const { retirement } = ages;
            if (retirement < current + 1) {
                throw new Validator_1.Validator.Failed('Retirement age should be at least one year after current age.');
            }
            if (max < retirement + 1) {
                throw new Validator_1.Validator.Failed('Max age should be at least one year after retirement age.');
            }
        }
        return ages;
    });
    const _person = object({
        ages: _ages,
        displayName: union(strBoundedTrimmed(TPAWParamsV12.MAX_LABEL_LENGTH), constant(null)),
    });
    const _year = union(object({ type: constant('now') }), object({
        type: constant('namedAge'),
        person: union(constant('person1'), constant('person2')),
        age: union(constant('lastWorkingYear'), constant('retirement'), constant('max')),
    }), object({
        type: constant('numericAge'),
        person: union(constant('person1'), constant('person2')),
        age: numInt(),
    }));
    const _yearRange = union(object({ type: constant('startAndEnd'), start: _year, end: _year }), object({
        type: constant('startAndNumYears'),
        start: _year,
        numYears: numIntPos(),
    }), object({
        type: constant('endAndNumYears'),
        end: _year,
        numYears: numIntPos(),
    }));
    const _valueForYearRange = object({
        // Not trimmed because it won't allow space even temporarily.
        label: union(strBounded(TPAWParamsV12.MAX_LABEL_LENGTH), constant(null)),
        yearRange: _yearRange,
        value: numGE(0),
        nominal: boolean(),
        id: numIntNonNeg(),
    });
    const _glidePath = object({
        start: object({ stocks: numRangeInclusive(0, 1) }),
        intermediate: sizedArray(object({ year: _year, stocks: numRangeInclusive(0, 1) }), 0, TPAWParamsV12.MAX_NUM_YEARS_IN_GLIDE_PATH - 2),
        end: object({ stocks: numRangeInclusive(0, 1) }),
    });
    TPAWParamsV12.validator = chain(object({
        v: constant(12),
        strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
        people: union(object({
            withPartner: constant(false),
            person1: _person,
        }), object({
            withPartner: constant(true),
            person2: _person,
            person1: _person,
            withdrawalStart: union(constant('person1'), constant('person2')),
            xAxis: union(constant('person1'), constant('person2')),
        })),
        returns: object({
            expected: union(object({ type: constant('suggested') }), object({ type: constant('oneOverCAPE') }), object({ type: constant('regressionPrediction') }), object({ type: constant('historical') }), object({
                type: constant('manual'),
                stocks: numRangeInclusive(-0.01, 0.1),
                bonds: numRangeInclusive(-0.01, 0.1),
            })),
            historical: union(object({
                type: constant('default'),
                adjust: union(object({ type: constant('to'), stocks: number(), bonds: number() }), object({ type: constant('by'), stocks: number(), bonds: number() }), object({ type: constant('toExpected') }), object({ type: constant('none') })),
            }), object({ type: constant('fixed'), stocks: number(), bonds: number() })),
        }),
        inflation: union(object({ type: constant('suggested') }), object({ type: constant('manual'), value: numRangeInclusive(-0.01, 0.1) })),
        targetAllocation: object({
            regularPortfolio: object({
                forTPAW: _glidePath,
                forSPAWAndSWR: _glidePath,
            }),
            legacyPortfolio: object({
                stocks: numRangeInclusive(0, 1),
            }),
        }),
        swrWithdrawal: union(object({ type: constant('asPercent'), percent: numRangeInclusive(0, 1) }), object({ type: constant('asAmount'), amount: numIntNonNeg() }), object({ type: constant('default') })),
        scheduledWithdrawalGrowthRate: numRangeInclusive(-0.03, 0.03),
        savingsAtStartOfStartYear: numGE(0),
        savings: array(_valueForYearRange),
        retirementIncome: array(_valueForYearRange),
        spendingCeiling: union(constant(null), numGE(0)),
        spendingFloor: union(constant(null), numGE(0)),
        withdrawals: object({
            lmp: numGE(0),
            essential: array(_valueForYearRange),
            discretionary: array(_valueForYearRange),
        }),
        legacy: object({
            total: numGE(0),
            external: array(object({
                label: union(string(), constant(null)),
                value: numGE(0),
                nominal: boolean(),
            })),
        }),
        sampling: union(constant('monteCarlo'), constant('historical')),
        display: object({ alwaysShowAllYears: boolean() }),
    }), x => {
        var _a;
        if (x.spendingCeiling !== null &&
            ((_a = x.spendingFloor) !== null && _a !== void 0 ? _a : 0) > x.spendingCeiling) {
            throw new Validator_1.Validator.Failed('Spending Floor is greater than spending ceiling.');
        }
        const checkYear = (year, prefix) => {
            if (year.type === 'namedAge' || year.type === 'numericAge') {
                let person;
                if (year.person === 'person1') {
                    person = x.people.person1;
                }
                else {
                    if (!x.people.withPartner) {
                        throw new Validator_1.Validator.Failed(`${prefix} is in terms of the age of an unspecified person.`);
                    }
                    person = x.people.person2;
                }
                if ((year.age === 'retirement' || year.age === 'lastWorkingYear') &&
                    person.ages.type === 'retired') {
                    throw new Validator_1.Validator.Failed(`${prefix} is in terms retirement age of ${year.person}, but ${year.person} is already retired.`);
                }
            }
        };
        const checkYearRange = (desc) => ({ yearRange, label }, i) => {
            if (yearRange.type === 'startAndEnd' ||
                yearRange.type === 'startAndNumYears') {
                checkYear(yearRange.start, `Starting year of ${desc} entry ${label !== null && label !== void 0 ? label : `at index ${i}`}`);
            }
            if (yearRange.type === 'startAndEnd' ||
                yearRange.type === 'endAndNumYears') {
                checkYear(yearRange.end, `Ending year of ${desc} ${label !== null && label !== void 0 ? label : `at index ${i}`}`);
            }
        };
        x.savings.forEach(checkYearRange('savings'));
        x.retirementIncome.forEach(checkYearRange('retirement income'));
        x.withdrawals.essential.forEach(checkYearRange('withdrawals funded by bonds'));
        x.withdrawals.discretionary.forEach(checkYearRange('withdrawals funded by risk portfolio'));
        const checkGlidePathYears = (glidePath) => {
            glidePath.forEach((x, i) => checkYear(x.year, `Year at entry ${i} in the static glide path`));
        };
        checkGlidePathYears(x.targetAllocation.regularPortfolio.forSPAWAndSWR.intermediate);
        return x;
    });
})(TPAWParamsV12 = exports.TPAWParamsV12 || (exports.TPAWParamsV12 = {}));
//# sourceMappingURL=TPAWParamsV12.js.map