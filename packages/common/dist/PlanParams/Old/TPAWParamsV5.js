"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.TPAWParamsV5 = void 0;
const lodash_1 = __importDefault(require("lodash"));
const Validator_1 = require("./Validator");
var TPAWParamsV5;
(function (TPAWParamsV5) {
    TPAWParamsV5.MAX_LABEL_LENGTH = 150;
    TPAWParamsV5.MAX_AGE = 120;
    const { number, string, constant, chain, object, union, array, boolean } = Validator_1.Validator;
    const numRangeInclusive = (min, max) => chain(number(), _geTest(min), _leTest(max));
    const numGE = (min) => chain(number(), _geTest(min));
    const numInt = () => chain(number(), _intTest);
    const numIntNonNeg = () => chain(numInt(), _geTest(0));
    const numIntPos = () => chain(numInt(), _geTest(1));
    const strBounded = (maxLength) => chain(string(), _boundedStringTest(maxLength));
    const strBoundedTrimmed = (maxLength) => chain(strBounded(maxLength), _trimmedTest);
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
    // ----------- VALIDATOR  ---------//
    const _ageRange = chain(number(), _geTest(0), _leTest(TPAWParamsV5.MAX_AGE));
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
        displayName: union(strBoundedTrimmed(TPAWParamsV5.MAX_LABEL_LENGTH), constant(null)),
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
        label: union(strBounded(TPAWParamsV5.MAX_LABEL_LENGTH), constant(null)),
        yearRange: _yearRange,
        value: numGE(0),
        nominal: boolean(),
        id: numIntNonNeg(),
    });
    TPAWParamsV5.validator = chain(object({
        v: constant(5),
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
            expected: object({
                stocks: numRangeInclusive(-0.01, 0.1),
                bonds: numRangeInclusive(-0.01, 0.1),
            }),
        }),
        inflation: numRangeInclusive(-0.01, 0.1),
        targetAllocation: object({
            regularPortfolio: object({
                stocks: numRangeInclusive(0, 1),
            }),
            legacyPortfolio: object({
                stocks: numRangeInclusive(0, 1),
            }),
        }),
        scheduledWithdrawalGrowthRate: numRangeInclusive(-0.03, 0.03),
        savingsAtStartOfStartYear: numGE(0),
        savings: array(_valueForYearRange),
        retirementIncome: array(_valueForYearRange),
        spendingCeiling: union(constant(null), numGE(0)),
        spendingFloor: union(constant(null), numGE(0)),
        withdrawals: object({
            fundedByBonds: array(_valueForYearRange),
            fundedByRiskPortfolio: array(_valueForYearRange),
        }),
        legacy: object({
            total: numGE(0),
            external: array(object({
                label: union(string(), constant(null)),
                value: numGE(0),
                nominal: boolean(),
            })),
        }),
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
                if (year.age === 'retirement' && person.ages.type === 'retired') {
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
        x.withdrawals.fundedByBonds.forEach(checkYearRange('withdrawals funded by bonds'));
        x.withdrawals.fundedByRiskPortfolio.forEach(checkYearRange('withdrawals funded by risk portfolio'));
        return x;
    });
})(TPAWParamsV5 = exports.TPAWParamsV5 || (exports.TPAWParamsV5 = {}));
//# sourceMappingURL=TPAWParamsV5.js.map