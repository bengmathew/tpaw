"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.TPAWParamsV4 = void 0;
const lodash_1 = __importDefault(require("lodash"));
const Validator_1 = require("./Validator");
var TPAWParamsV4;
(function (TPAWParamsV4) {
    const { number, string, constant, chain, object, union, array, boolean } = Validator_1.Validator;
    const numRangeInclusive = (min, max) => chain(number(), _geTest(min), _leTest(max));
    const numGE = (min) => chain(number(), _geTest(min));
    const numNonNegInteger = () => chain(number(), _intTest, _geTest(0));
    const _intTest = (x) => {
        if (!lodash_1.default.isInteger(x))
            throw new Validator_1.Validator.Failed('Not an integer.');
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
    // ----------- VALIDATOR ---------//
    TPAWParamsV4.MAX_AGE_V4 = 120;
    const _ageRange = chain(number(), _geTest(0), _leTest(TPAWParamsV4.MAX_AGE_V4));
    const _ageValidator = chain(object({
        start: _ageRange,
        retirement: _ageRange,
        end: _ageRange,
    }), (age) => {
        const { start, retirement, end } = age;
        if (retirement < start) {
            throw new Validator_1.Validator.Failed('Retirement age is earlier than current age.');
        }
        if (end < start + 1) {
            throw new Validator_1.Validator.Failed('Max age should be at least one year after current age.');
        }
        if (retirement >= end) {
            throw new Validator_1.Validator.Failed('Max age should be at least one year after retirement age.');
        }
        return age;
    });
    const _yearRangeEdge = union(number(), constant('start'), constant('lastWorkingYear'), constant('retirement'), constant('end'));
    const _valueForYearRangeValidator = object({
        label: union(string(), constant(null)),
        yearRange: object({ start: _yearRangeEdge, end: _yearRangeEdge }),
        value: numGE(0),
        nominal: boolean(),
        id: numNonNegInteger(),
    });
    TPAWParamsV4.validator = chain(object({
        v: constant(4),
        age: _ageValidator,
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
        savings: array(_valueForYearRangeValidator),
        retirementIncome: array(_valueForYearRangeValidator),
        spendingCeiling: union(constant(null), numGE(0)),
        spendingFloor: union(constant(null), numGE(0)),
        withdrawals: object({
            fundedByBonds: array(_valueForYearRangeValidator),
            fundedByRiskPortfolio: array(_valueForYearRangeValidator),
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
            throw new Validator_1.Validator.Failed('Spending Floor is greater than spending ceiling');
        }
        return x;
    });
})(TPAWParamsV4 = exports.TPAWParamsV4 || (exports.TPAWParamsV4 = {}));
//# sourceMappingURL=TPAWParamsV4.js.map