import {
  array,
  boolean,
  bounded,
  chain,
  constant,
  failure,
  gt,
  gte,
  integer,
  JSONGuard,
  JSONGuardResult,
  lte,
  nullable,
  number,
  object,
  string,
  success,
  union,
} from 'json-guard'
import _ from 'lodash'
import {
  fGet,
  linearFnFomPoints,
  linearFnFromPointAndSlope,
  noCase,
  preciseRange,
} from '../../Utils'
import { PlanParams14 } from './PlanParams14'

export namespace PlanParams15 {
  export const MAX_LABEL_LENGTH = 150
  export const MAX_AGE = 120
  export const MAX_NUM_YEARS_IN_GLIDE_PATH = 1000
  export const MAX_VALUE_FOR_YEAR_RANGE = 100
  export const MAX_EXTERNAL_LEGACY_SOURCES = 100
  export const TIME_PREFERENCE_VALUES = preciseRange(-0.05, 0.05, 0.001, 3)
  export const MANUAL_INFLATION_VALUES = preciseRange(-0.01, 0.1, 0.001, 3)
  export const MANUAL_STOCKS_BONDS_RETURNS_VALUES = preciseRange(
    -0.01,
    0.1,
    0.001,
    3,
  )

  export const SPAW_SPENDING_TILT_VALUES = preciseRange(-0.03, 0.03, 0.001, 3)
  export const RISK_TOLERANCE_VALUES = (() => {
    const numSegments = 5
    const countPerSegment = 5
    const numPoints = numSegments * countPerSegment
    const startRRA = 8
    const endRRA = 0.25

    const { rraToRiskTolerance, riskToleranceToRRA } = (() => {
      const log1OverRRA = (rra: number) => Math.log(1 / rra)
      const shift = log1OverRRA(startRRA)
      const scale =
        (numPoints - 2) / (log1OverRRA(endRRA) - log1OverRRA(startRRA))

      const rraToRiskTolerance = (rra: number) =>
        (log1OverRRA(rra) - shift) * scale + 1

      const riskToleranceToRRA = (riskTolerance: number) =>
        1 / Math.exp((riskTolerance - 1) / scale + shift)

      return { rraToRiskTolerance, riskToleranceToRRA }
    })()

    const DATA = _.times(numPoints, (i) => i)

    const segmentDef = (segment: number, label: string) => {
      const startIndex = segment * numSegments
      const endIndex = startIndex + countPerSegment - 1 // end is inclusive
      const count = countPerSegment
      const containsIndex = (index: number) =>
        index >= startIndex && index <= endIndex
      return { startIndex, endIndex, containsIndex, label, count }
    }

    const SEGMENTS = [
      segmentDef(0, 'Very Conservative'),
      segmentDef(1, 'Conservative'),
      segmentDef(2, 'Moderate'),
      segmentDef(3, 'Aggressive'),
      segmentDef(4, 'Very Aggressive'),
    ]
    return { DATA, SEGMENTS, riskToleranceToRRA, rraToRiskTolerance }
  })()

  export type Person = {
    ages:
      | { type: 'retired'; current: number; max: number }
      | { type: 'notRetired'; current: number; retirement: number; max: number }
    displayName: string | null
  }

  export type People =
    | { withPartner: false; person1: Person }
    | {
        withPartner: true
        person2: Person
        person1: Person
        withdrawalStart: 'person1' | 'person2'
        xAxis: 'person1' | 'person2'
      }

  export type Year =
    | { type: 'now' }
    | {
        type: 'namedAge'
        person: 'person1' | 'person2'
        age: 'lastWorkingYear' | 'retirement' | 'max'
      }
    | {
        type: 'numericAge'
        person: 'person1' | 'person2'
        age: number
      }

  export type YearRange =
    | { type: 'startAndEnd'; start: Year; end: Year }
    | { type: 'startAndNumYears'; start: Year; numYears: number }
    | { type: 'endAndNumYears'; end: Year; numYears: number }

  export type ValueForYearRange = {
    label: string | null
    yearRange: YearRange
    value: number
    nominal: boolean
    id: number
  }

  export type LabeledAmount = {
    label: string | null
    value: number
    nominal: boolean
  }

  export type GlidePath = {
    start: { stocks: number }
    intermediate: { year: Year; stocks: number }[]
    end: { stocks: number }
  }

  export type Params = {
    v: 15
    warnedAbout14to15Converstion: boolean
    strategy: 'TPAW' | 'SPAW' | 'SWR'
    dialogMode: boolean

    // Basic Inputs
    people: People
    currentPortfolioBalance: number
    futureSavings: ValueForYearRange[]
    retirementIncome: ValueForYearRange[]

    // Adjustments to Spending
    adjustmentsToSpending: {
      tpawAndSPAW: {
        spendingCeiling: number | null
        spendingFloor: number | null
        legacy: {
          total: number
          external: LabeledAmount[]
        }
      }
      extraSpending: {
        essential: ValueForYearRange[]
        discretionary: ValueForYearRange[]
      }
    }

    // Risk
    risk: {
      tpaw: {
        riskTolerance: {
          at20: number
          deltaAtMaxAge: number
          forLegacyAsDeltaFromAt20: number
        }
        timePreference: number
      }
      tpawAndSPAW: {
        lmp: number
      }
      spaw: {
        spendingTilt: number
      }
      spawAndSWR: {
        allocation: GlidePath
      }
      swr: {
        withdrawal:
          | { type: 'asPercent'; percent: number }
          | { type: 'asAmount'; amount: number }
          | { type: 'default' }
      }
    }

    // Advanced.
    returns: {
      expected:
        | { type: 'suggested' }
        | { type: 'oneOverCAPE' }
        | { type: 'regressionPrediction' }
        | { type: 'historical' }
        | { type: 'manual'; stocks: number; bonds: number }
      historical:
        | {
            type: 'default'
            adjust:
              | { type: 'by'; stocks: number; bonds: number }
              | { type: 'to'; stocks: number; bonds: number }
              | { type: 'toExpected' }
              | { type: 'none' }
          }
        | { type: 'fixed'; stocks: number; bonds: number }
    }
    inflation: { type: 'suggested' } | { type: 'manual'; value: number }
    sampling: 'monteCarlo' | 'historical'

    // Other.
    display: {
      alwaysShowAllYears: boolean
    }
  }

  // ----------- GUARD  ---------//

  const among =
    <T>(values: T[]): JSONGuard<T> =>
    (x: unknown) => {
      return values.includes(x as T)
        ? success(x as T)
        : failure('Not among predefined value.')
    }

  const _ageRange = chain(number, gte(0), lte(MAX_AGE))

  const _ages: JSONGuard<Person['ages']> = chain(
    union(
      object({
        type: constant('retired'),
        current: _ageRange,
        max: _ageRange,
      }),
      object({
        type: constant('notRetired'),
        current: _ageRange,
        retirement: _ageRange,
        max: _ageRange,
      }),
    ),
    (ages: Person['ages']): JSONGuardResult<Person['ages']> => {
      const { current, max } = ages
      if (max < current + 1) {
        return failure('Max age should be at least one year after current age.')
      }
      if (ages.type === 'notRetired') {
        const { retirement } = ages
        if (retirement < current + 1) {
          return failure(
            'Retirement age should be at least one year after current age.',
          )
        }
        if (max < retirement + 1) {
          return failure(
            'Max age should be at least one year after retirement age.',
          )
        }
      }
      return success(ages)
    },
  )

  const person: JSONGuard<Person> = object({
    ages: _ages,
    displayName: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
  })

  const people: JSONGuard<Params['people']> = union(
    object({
      withPartner: constant(false),
      person1: person,
    }),
    object({
      withPartner: constant(true),
      person2: person,
      person1: person,
      withdrawalStart: union(constant('person1'), constant('person2')),
      xAxis: union(constant('person1'), constant('person2')),
    }),
  )

  const year = (people: Params['people'] | null) =>
    union(
      object({ type: constant('now') }),
      chain(
        object({
          type: constant('namedAge'),
          person: union(constant('person1'), constant('person2')),
          age: union(
            constant('lastWorkingYear'),
            constant('retirement'),
            constant('max'),
          ),
        }),
        (x) => {
          if (!people) return success(x)
          let person: Person
          if (x.person === 'person1') {
            person = people.person1
          } else {
            if (!people.withPartner)
              return failure('In terms of partner, but there is no partner.')
            person = people.person2
          }
          if (
            (x.age === 'retirement' || x.age === 'lastWorkingYear') &&
            person.ages.type === 'retired'
          ) {
            return failure(
              `In terms retirement age of ${x.person}, but ${x.person} is already retired.`,
            )
          }
          return success(x)
        },
      ),
      chain(
        object({
          type: constant('numericAge'),
          person: union(constant('person1'), constant('person2')),
          age: chain(number, integer),
        }),
        (x) => {
          if (!people) return success(x)
          if (x.person === 'person2' && !people.withPartner)
            return failure('In terms of partner, but there is no partner.')
          return success(x)
        },
      ),
    )

  const yearRange = (people: Params['people'] | null): JSONGuard<YearRange> => {
    const yr = year(people)
    return union(
      object({ type: constant('startAndEnd'), start: yr, end: yr }),
      object({
        type: constant('startAndNumYears'),
        start: yr,
        numYears: chain(number, integer, gt(0)),
      }),
      object({
        type: constant('endAndNumYears'),
        end: yr,
        numYears: chain(number, integer, gt(0)),
      }),
    )
  }

  const valueForYearRange = (
    people: Params['people'] | null,
  ): JSONGuard<ValueForYearRange> =>
    object({
      // Not trimmed because it won't allow space even temporarily.
      label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
      yearRange: yearRange(people),
      value: chain(number, gte(0)),
      nominal: boolean,
      id: chain(number, integer, gte(0)),
    })
  const valueForYearRangeArr = (
    people: Params['people'] | null = null,
  ): JSONGuard<ValueForYearRange[]> =>
    array(valueForYearRange(people), MAX_VALUE_FOR_YEAR_RANGE)

  const adjustmentsToSpending = (
    params: Params | null,
  ): JSONGuard<Params['adjustmentsToSpending']> =>
    object({
      tpawAndSPAW: object({
        spendingCeiling: chain(
          nullable(chain(number, gte(0))),
          chain(nullable(chain(number, gte(0))), (x) => {
            if (!params) return success(x)
            if (x === null) return success(x)
            if (
              params.adjustmentsToSpending.tpawAndSPAW.spendingFloor !== null &&
              x < params.adjustmentsToSpending.tpawAndSPAW.spendingFloor
            ) {
              failure('Spending Floor is greater than spending ceiling.')
            }
            return success(x)
          }),
        ),
        spendingFloor: nullable(chain(number, gte(0))),
        legacy: object({
          total: chain(number, gte(0)),
          external: array(
            object({
              label: nullable(chain(string, bounded(MAX_LABEL_LENGTH))),
              value: chain(number, gte(0)),
              nominal: boolean,
            }),
            MAX_EXTERNAL_LEGACY_SOURCES,
          ),
        }),
      }),
      extraSpending: object({
        essential: array(
          valueForYearRange(params?.people ?? null),
          MAX_VALUE_FOR_YEAR_RANGE,
        ),
        discretionary: array(
          valueForYearRange(params?.people ?? null),
          MAX_VALUE_FOR_YEAR_RANGE,
        ),
      }),
    })

  const glidePath = (params: Params | null): JSONGuard<GlidePath> =>
    object({
      start: object({ stocks: chain(number, gte(0), lte(1)) }),
      intermediate: array(
        object({
          year: year(params?.people ?? null),
          stocks: chain(number, gte(0), lte(1)),
        }),
        MAX_NUM_YEARS_IN_GLIDE_PATH - 2,
      ),
      end: object({ stocks: chain(number, gte(0), lte(1)) }),
    })

  const risk = (params: Params | null): JSONGuard<Params['risk']> =>
    object({
      tpaw: object({
        riskTolerance: object({
          at20: among(RISK_TOLERANCE_VALUES.DATA),
          deltaAtMaxAge: among(RISK_TOLERANCE_VALUES.DATA.map((x) => x * -1)),
          forLegacyAsDeltaFromAt20: among(RISK_TOLERANCE_VALUES.DATA),
        }),
        timePreference: among(TIME_PREFERENCE_VALUES),
      }),
      tpawAndSPAW: object({
        lmp: chain(number, gte(0)),
      }),
      spaw: object({ spendingTilt: among(SPAW_SPENDING_TILT_VALUES) }),
      swr: object({
        withdrawal: union(
          object({
            type: constant('asPercent'),
            percent: chain(number, gte(0), lte(1)),
          }),
          object({
            type: constant('asAmount'),
            amount: chain(number, integer, gte(0)),
          }),
          object({ type: constant('default') }),
        ),
      }),
      spawAndSWR: object({
        allocation: glidePath(params),
      }),
    })

  const returns: JSONGuard<Params['returns']> = object({
    expected: union(
      object({ type: constant('suggested') }),
      object({ type: constant('oneOverCAPE') }),
      object({ type: constant('regressionPrediction') }),
      object({ type: constant('historical') }),
      object({
        type: constant('manual'),
        stocks: among(MANUAL_STOCKS_BONDS_RETURNS_VALUES),
        bonds: among(MANUAL_STOCKS_BONDS_RETURNS_VALUES),
      }),
    ),
    historical: union(
      object({
        type: constant('default'),
        adjust: union(
          object({
            type: constant('to'),
            stocks: number,
            bonds: number,
          }),
          object({
            type: constant('by'),
            stocks: number,
            bonds: number,
          }),
          object({ type: constant('toExpected') }),
          object({ type: constant('none') }),
        ),
      }),
      object({
        type: constant('fixed'),
        stocks: number,
        bonds: number,
      }),
    ),
  })

  const inflation: JSONGuard<Params['inflation']> = union(
    object({ type: constant('suggested') }),
    object({
      type: constant('manual'),
      value: among(MANUAL_INFLATION_VALUES),
    }),
  )

  const params = (x: Params | null): JSONGuard<Params> =>
    object({
      v: constant(15),
      warnedAbout14to15Converstion: boolean,
      strategy: union(constant('TPAW'), constant('SPAW'), constant('SWR')),
      dialogMode: boolean,
      people,
      currentPortfolioBalance: chain(number, gte(0)),
      futureSavings: valueForYearRangeArr(x?.people ?? null),
      retirementIncome: valueForYearRangeArr(x?.people ?? null),
      adjustmentsToSpending: adjustmentsToSpending(x),
      risk: risk(x),
      returns,
      inflation,
      sampling: union(constant('monteCarlo'), constant('historical')),
      display: object({ alwaysShowAllYears: boolean }),
    })

  const v15Guard: JSONGuard<Params> = chain(params(null), (x) => params(x)(x))


  export const guard: JSONGuard<Params> = (x: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-explicit-any
    if ((x as any).v === 15) {
      return v15Guard(x)
    } else {
      const v14Result = PlanParams14.guard(x)
      if (v14Result.error) return v14Result
      const { extraSpending, legacy, risk, ...v14 } = v14Result.value

      const adjustmentsToSpending = {
        tpawAndSPAW: (() => {
          return {
            spendingCeiling: risk.useTPAWPreset
              ? null
              : risk.tpawAndSPAW.spendingCeiling,
            spendingFloor: risk.useTPAWPreset
              ? null
              : risk.tpawAndSPAW.spendingFloor,
            legacy: legacy.tpawAndSPAW,
          }
        })(),
        extraSpending: extraSpending,
      }

      const oldRisk = FOR14.resolveTPAWRiskPreset(
        risk,
        FOR14.getNumYears(v14.people),
      )

      const tpawRisk = risk.useTPAWPreset
        ? (() => {
            const result = (at20: number): Params['risk']['tpaw'] => ({
              riskTolerance: {
                at20,
                deltaAtMaxAge: -2,
                forLegacyAsDeltaFromAt20: 2,
              },
              timePreference: 0,
            })
            switch (risk.tpawPreset) {
              case 'riskLevel-1':
                return result(9)
              case 'riskLevel-2':
                return result(12)
              case 'riskLevel-3':
                return result(14)
              case 'riskLevel-4':
                return result(16)
              default:
                noCase(risk.tpawPreset)
            }
          })()
        : FOR14.v14To15TPAWRisk(
            oldRisk.tpaw.allocation.start.stocks,
            oldRisk.tpawAndSPAW.spendingTilt,
            oldRisk.tpaw.allocationForLegacy.stocks,
            v14.returns,
            v14.people,
          )

      const newRisk: Params['risk'] = {
        tpaw: tpawRisk,
        tpawAndSPAW: {
          lmp: oldRisk.tpawAndSPAW.lmp,
        },
        spaw: {
          spendingTilt: oldRisk.tpawAndSPAW.spendingTilt,
        },
        spawAndSWR: {
          allocation: oldRisk.spawAndSWR.allocation,
        },
        swr: {
          withdrawal: oldRisk.swr.withdrawal,
        },
      }

      const v15: Params = {
        ...v14,
        v: 15,
        warnedAbout14to15Converstion: false,
        adjustmentsToSpending,
        risk: newRisk,
      }
      return success(v15Guard(v15).force())
    }
  }

  // --------------------------------------------------------
  // -------------------- FOR 14 ----------------------------
  // --------------------------------------------------------
  namespace FOR14 {
    export const v14To15TPAWRisk = (
      startingStockAllocation: number,
      spendingTilt: number,
      allocationForLegacy: number,
      v14Returns: PlanParams14.Params['returns'],
      v14People: PlanParams14.Params['people'],
    ): Params['risk']['tpaw'] => {
      // console.dir('--------------')
      // console.dir(`Converting`)
      // console.dir(`    staringStockAllocation:${startingStockAllocation}`)
      // console.dir(`    spendingTilt:${spendingTilt}`)
      // console.dir(`    allocationForLegacy:${allocationForLegacy}`)

      const tpawRiskNow =
        FOR14.stockAllocationAndSpendingTiltToRiskToleranceAndTimePreference(
          startingStockAllocation,
          spendingTilt,
          v14Returns,
        )

      // console.dir(`riskNow`)
      // console.dir(`    riskTolerance: ${tpawRiskNow.riskTolerance}`)
      // console.dir(`    timePreference: ${tpawRiskNow.timePreference}`)
      // console.dir('    WHich gives')
      // console.dir(`        stockAllocation: ${tpawRiskNow.stockAllocation}`)
      // console.dir(`        spendingTilt: ${tpawRiskNow.spendingTilt}`)

      const deltaAtMaxAge = -2

      const roundAndClampRiskTolerance = (x: number) =>
        _.clamp(
          Math.round(x),
          fGet(_.first(RISK_TOLERANCE_VALUES.DATA)),
          fGet(_.last(RISK_TOLERANCE_VALUES.DATA)),
        )

      const mergedAge = (() => {
        const { person1 } = v14People
        if (!v14People.withPartner) return person1.ages
        const { person2 } = v14People
        const person1LifeSpan = person1.ages.max - person1.ages.current
        const person2LifeSpan = person2.ages.max - person2.ages.current
        return person2LifeSpan > person1LifeSpan ? person2.ages : person1.ages
      })()
      // console.dir(`Age`)
      // console.dir(`    current: ${mergedAge.current}`)
      // console.dir(`    max: ${mergedAge.max}`)
      const at20 = roundAndClampRiskTolerance(
        linearFnFromPointAndSlope(
          mergedAge.current,
          tpawRiskNow.riskTolerance,
          deltaAtMaxAge / 80,
        )(20),
      )
      const forLegacy =
        FOR14.stockAllocationAndSpendingTiltToRiskToleranceAndTimePreference(
          allocationForLegacy,
          0, // Does not matter.
          v14Returns,
        )

      const forLegacyAsDeltaFromAt20 = roundAndClampRiskTolerance(
        forLegacy.riskTolerance - at20,
      )

      // console.dir('Legacy')
      // console.dir(`    riskTolerance : ${forLegacy.riskTolerance}`)
      // console.dir(`    timePreference: ${forLegacy.timePreference}`)
      // console.dir('    WHich gives')
      // console.dir(`        stockAllocation: ${forLegacy.stockAllocation}`)
      // console.dir(`        spendingTilt: ${forLegacy.spendingTilt}`)
      // console.dir(`    asDeltaFrom20: ${forLegacyAsDeltaFromAt20}`)

      // console.dir('RESUlTS:')
      // console.dir(`    riskToleranceAt20: ${at20}`)
      // console.dir(`    deltaAtMaxAge: ${deltaAtMaxAge}`)
      // console.dir(`    timePreference: ${tpawRiskNow.timePreference}`)
      // console.dir(`    forLegacyAsDeltaFromAt20: ${forLegacyAsDeltaFromAt20}`)

      return {
        riskTolerance: {
          at20,
          deltaAtMaxAge,
          forLegacyAsDeltaFromAt20,
        },
        timePreference: tpawRiskNow.timePreference,
      }
    }

    export const getNumYears = (people: PlanParams14.Params['people']) => {
      const forPerson = (person: PlanParams14.Person) =>
        person.ages.max - person.ages.current + 1
      return Math.max(
        forPerson(people.person1),
        people.withPartner ? forPerson(people.person2) : 0,
      )
    }
    export const TPAW_RISK_PRESETS = (
      preset: Exclude<PlanParams14.TPAWRiskLevel, 'custom'>,
      numYears: number,
    ): PlanParams14.TPAWRisk => {
      const allocFn = (
        at20of100: number,
        spendingTilt: number,
      ): PlanParams14.TPAWRisk => {
        const at100of100 = _.round(at20of100 - 0.1, 2)
        return {
          tpaw: {
            allocation: {
              start: {
                stocks: _.round(
                  linearFnFomPoints(
                    0,
                    at100of100,
                    100 - 20,
                    at20of100,
                  )(numYears),
                  2,
                ),
              },
              intermediate: [],
              end: { stocks: at100of100 },
            },
            allocationForLegacy: { stocks: _.round(at20of100 + 0.2, 2) },
          },
          tpawAndSPAW: {
            spendingCeiling: null,
            spendingFloor: null,
            spendingTilt,
            lmp: 0,
          },
        }
      }
      switch (preset) {
        case 'riskLevel-1':
          return allocFn(0.3, 0.005)
        case 'riskLevel-2':
          return allocFn(0.4, 0.008)
        case 'riskLevel-3':
          return allocFn(0.5, 0.011)
        case 'riskLevel-4':
          return allocFn(0.6, 0.014)
        default:
          noCase(preset)
      }
    }
    export const resolveTPAWRiskPreset = (
      risk: PlanParams14.Params['risk'],
      numYears: number,
    ): Exclude<PlanParams14.Params['risk'], { useTPAWPreset: true }> => {
      if (!risk.useTPAWPreset) return risk
      const { swr, spawAndSWR, tpawPreset, customTPAWPreset, savedTPAWPreset } =
        risk
      const { tpaw, tpawAndSPAW } = TPAW_RISK_PRESETS(risk.tpawPreset, numYears)
      return {
        useTPAWPreset: false,
        tpaw,
        tpawAndSPAW,
        swr,
        spawAndSWR,
        tpawPreset,
        customTPAWPreset,
        savedTPAWPreset,
      }
    }

    export const stockAllocationAndSpendingTiltToRiskToleranceAndTimePreference =
      (
        stockAllocation: number,
        spendingTilt: number,
        returns14: PlanParams14.Params['returns'],
      ) => {
        const returnsProcessed = _processReturnsParams(returns14)
        const table = _.flatten(
          RISK_TOLERANCE_VALUES.DATA.map((riskTolerance) =>
            TIME_PREFERENCE_VALUES.map((timePreference) => {
              const { gContinuous, stockAllocation } = _applyMerton(
                returnsProcessed,
                riskTolerance,
                timePreference,
              )
              const spendingTilt = Math.exp(gContinuous) - 1
              return {
                riskTolerance,
                timePreference,
                stockAllocation,
                spendingTilt,
              }
            }),
          ),
        )
        return fGet(
          _.minBy(table, (x) =>
            Math.sqrt(
              Math.pow(stockAllocation - x.stockAllocation, 2) +
                Math.pow(spendingTilt - x.spendingTilt, 2),
            ),
          ),
        )
      }

    export const _applyMerton = (
      returns: ReturnType<typeof _processReturnsParams>,
      riskTolerance: number,
      timePreference: number,
    ) => {
      if (riskTolerance === 0) {
        return {
          gContinuous: timePreference,
          stockAllocation: 0,
        }
      }
      const rho = timePreference
      // Don't use historicalReturns.stocks.convertExpectedToLog() because we
      // want to assume variance is 0 and not use the historical variance,
      // unlike for stocks.
      const r = Math.log(1 + returns.expected.bonds)
      const mu = historicalReturns.stocks.convertExpectedToLog(
        returns.expected.stocks,
      )
      const sigmaPow2 = historicalReturns.stocks.log.variance
      const gamma = RISK_TOLERANCE_VALUES.riskToleranceToRRA(riskTolerance)

      const stockAllocation = Math.min(1, (mu - r) / (sigmaPow2 * gamma))

      const nu =
        (rho -
          (1 - gamma) * (Math.pow(mu - r, 2) / (2 * sigmaPow2 * gamma) + r)) /
        gamma

      const logROfPortfolio = historicalReturns.statsFn(
        returns.historicalAdjusted.map(
          ({ stocks }) =>
            stocks * stockAllocation +
            returns.expected.bonds * (1 - stockAllocation),
        ),
      ).log.expectedValue
      const gContinuous = logROfPortfolio - nu

      return { gContinuous, stockAllocation }
    }

    function _processReturnsParams(returns: PlanParams14.Params['returns']) {
      const expected = processExpectedReturns(returns.expected)

      const historicalAdjusted = (() => {
        switch (returns.historical.type) {
          case 'default': {
            const adjustment = returns.historical.adjust
            const adjust = (type: 'stocks' | 'bonds') => {
              const historical = historicalReturns[type]

              const targetExpected =
                adjustment.type === 'to'
                  ? adjustment[type]
                  : adjustment.type === 'toExpected'
                  ? expected[type]
                  : adjustment.type === 'none'
                  ? historical.expectedValue
                  : adjustment.type === 'by'
                  ? historical.expectedValue - adjustment[type]
                  : noCase(adjustment)

              return historical.adjust(targetExpected)
            }

            return _.zipWith(
              adjust('stocks'),
              adjust('bonds'),
              (stocks, bonds) => ({ stocks, bonds }),
            )
          }
          case 'fixed': {
            const { stocks, bonds } = returns.historical
            return _.times(historicalReturns.raw.length, () => ({
              stocks,
              bonds,
            }))
          }
          default:
            noCase(returns.historical)
        }
      })()

      return { historical: returns.historical, expected, historicalAdjusted }
    }
    function processExpectedReturns(
      expected: PlanParams14.Params['returns']['expected'],
    ) {
      switch (expected.type) {
        case 'manual':
          return { stocks: expected.stocks, bonds: expected.bonds }
        default:
          return EXPECTED_RETURN_PRESETS(expected.type)
      }
    }

    const marketData = {
      CAPE: {
        date: 1671436800000,
        value: 27.93,
        oneOverCAPE: 0.03580379520229144,
        regression: {
          full: {
            fiveYear: 0.047340656706045925,
            tenYear: 0.04792626638685493,
            twentyYear: 0.05606743070710185,
            thirtyYear: 0.06930417420093193,
          },
          restricted: {
            fiveYear: 0.06236223740404334,
            tenYear: 0.04964930793418443,
            twentyYear: 0.04727633572356149,
            thirtyYear: 0.06756688161109681,
          },
        },
        regressionAverage: 0.05593666133422759,
        suggested: 0.044586763504688445,
      },
      inflation: { date: 1671148800000, value: 0.0213 },
      bondRates: {
        date: 1671436800000,
        fiveYear: 0.0151,
        sevenYear: 0.0146,
        tenYear: 0.014199999999999999,
        twentyYear: 0.0143,
        thirtyYear: 0.0147,
      },
    }

    export const EXPECTED_RETURN_PRESETS = (
      type: Exclude<
        PlanParams14.Params['returns']['expected']['type'],
        'manual'
      >,
    ) => {
      const { CAPE, bondRates } = marketData
      const suggested = {
        stocks: _.round(CAPE.suggested, 3),
        bonds: _.round(bondRates.twentyYear, 3),
      }
      switch (type) {
        case 'suggested':
          return { ...suggested }
        case 'oneOverCAPE':
          return {
            stocks: _.round(CAPE.oneOverCAPE, 3),
            bonds: suggested.bonds,
          }
        case 'regressionPrediction':
          return {
            stocks: _.round(CAPE.regressionAverage, 3),
            bonds: suggested.bonds,
          }
        case 'historical':
          return {
            stocks: historicalReturns.stocks.expectedValue,
            bonds: historicalReturns.bonds.expectedValue,
          }
        default:
          noCase(type)
      }
    }

    export type MarketData = {
      CAPE: {
        suggested: number
        oneOverCAPE: number
        regressionAverage: number
      }
      bondRates: { twentyYear: number }
      inflation: { value: number }
    }

    const raw = [
      { year: 1871, stocks: 0.13905431, bonds: 0.035670483 },
      { year: 1872, stocks: 0.087811663, bonds: 0.015561673 },
      { year: 1873, stocks: 0.02034298, bonds: 0.114757246 },
      { year: 1874, stocks: 0.12500825, bonds: 0.167868952 },
      { year: 1875, stocks: 0.118238609, bonds: 0.156717926 },
      { year: 1876, stocks: -0.149164304, bonds: 0.048715308 },
      { year: 1877, stocks: 0.169730057, bonds: 0.2497398 },
      { year: 1878, stocks: 0.29615134, bonds: 0.174941611 },
      { year: 1879, stocks: 0.237991719, bonds: -0.122469893 },
      { year: 1880, stocks: 0.343651921, bonds: 0.13173245 },
      { year: 1881, stocks: -0.071933746, bonds: -0.033918711 },
      { year: 1882, stocks: 0.055871765, bonds: 0.055723236 },
      { year: 1883, stocks: 0.023113088, bonds: 0.123317381 },
      { year: 1884, stocks: -0.023055371, bonds: 0.165093382 },
      { year: 1885, stocks: 0.345350511, bonds: 0.085566342 },
      { year: 1886, stocks: 0.119399916, bonds: 0.022025203 },
      { year: 1887, stocks: -0.051459357, bonds: -0.022886134 },
      { year: 1888, stocks: 0.082091397, bonds: 0.105680177 },
      { year: 1889, stocks: 0.124177568, bonds: 0.089385937 },
      { year: 1890, stocks: -0.084472115, bonds: -0.006284823 },
      { year: 1891, stocks: 0.266030373, bonds: 0.105865312 },
      { year: 1892, stocks: -0.015188809, bonds: -0.049549462 },
      { year: 1893, stocks: -0.063908402, bonds: 0.201433581 },
      { year: 1894, stocks: 0.080576405, bonds: 0.103345999 },
      { year: 1895, stocks: 0.034592435, bonds: 0.009176227 },
      { year: 1896, stocks: 0.062552604, bonds: 0.084048523 },
      { year: 1897, stocks: 0.169339864, bonds: 0.008972849 },
      { year: 1898, stocks: 0.275202666, bonds: 0.040035116 },
      { year: 1899, stocks: -0.113094917, bonds: -0.121219722 },
      { year: 1900, stocks: 0.239383238, bonds: 0.061698511 },
      { year: 1901, stocks: 0.165856277, bonds: 0.000140906 },
      { year: 1902, stocks: -0.012315209, bonds: -0.067469381 },
      { year: 1903, stocks: -0.132751842, bonds: 0.072462283 },
      { year: 1904, stocks: 0.291338059, bonds: 0.004910918 },
      { year: 1905, stocks: 0.213054594, bonds: 0.039460539 },
      { year: 1906, stocks: -0.036322419, bonds: -0.02819569 },
      { year: 1907, stocks: -0.225169879, bonds: 0.043743077 },
      { year: 1908, stocks: 0.349742449, bonds: 0.014851759 },
      { year: 1909, stocks: 0.050158595, bonds: -0.07243636 },
      { year: 1910, stocks: 0.035937973, bonds: 0.108849708 },
      { year: 1911, stocks: 0.045985335, bonds: 0.04894225 },
      { year: 1912, stocks: -0.001044883, bonds: -0.06182729 },
      { year: 1913, stocks: -0.066429762, bonds: 0.04725387 },
      { year: 1914, stocks: -0.064058557, bonds: 0.025813065 },
      { year: 1915, stocks: 0.274338047, bonds: 0.02794096 },
      { year: 1916, stocks: -0.037917781, bonds: -0.087075486 },
      { year: 1917, stocks: -0.319210398, bonds: -0.150313463 },
      { year: 1918, stocks: 0.001762485, bonds: -0.107250048 },
      { year: 1919, stocks: 0.022727699, bonds: -0.136446996 },
      { year: 1920, stocks: -0.126189208, bonds: 0.058120394 },
      { year: 1921, stocks: 0.237614393, bonds: 0.25425432 },
      { year: 1922, stocks: 0.298966975, bonds: 0.045315199 },
      { year: 1923, stocks: 0.024094301, bonds: 0.037708143 },
      { year: 1924, stocks: 0.271164853, bonds: 0.057533791 },
      { year: 1925, stocks: 0.21653545, bonds: 0.018615772 },
      { year: 1926, stocks: 0.141147006, bonds: 0.089949952 },
      { year: 1927, stocks: 0.387451153, bonds: 0.046700985 },
      { year: 1928, stocks: 0.493477214, bonds: 0.023818508 },
      { year: 1929, stocks: -0.094269684, bonds: 0.062316808 },
      { year: 1930, stocks: -0.168951842, bonds: 0.106977499 },
      { year: 1931, stocks: -0.380278812, bonds: 0.119178749 },
      { year: 1932, stocks: 0.040196059, bonds: 0.184059272 },
      { year: 1933, stocks: 0.531039727, bonds: 0.025585512 },
      { year: 1934, stocks: -0.10710655, bonds: 0.028446701 },
      { year: 1935, stocks: 0.527106998, bonds: 0.025063804 },
      { year: 1936, stocks: 0.298931558, bonds: 0.002498737 },
      { year: 1937, stocks: -0.325606707, bonds: 0.030041385 },
      { year: 1938, stocks: 0.189489087, bonds: 0.058004215 },
      { year: 1939, stocks: 0.037984007, bonds: 0.044281711 },
      { year: 1940, stocks: -0.101714377, bonds: 0.030282826 },
      { year: 1941, stocks: -0.183369022, bonds: -0.12279712 },
      { year: 1942, stocks: 0.129713226, bonds: -0.048684514 },
      { year: 1943, stocks: 0.200691966, bonds: -0.005298894 },
      { year: 1944, stocks: 0.170022562, bonds: 0.011271917 },
      { year: 1945, stocks: 0.363003417, bonds: 0.016695187 },
      { year: 1946, stocks: -0.255345641, bonds: -0.139123032 },
      { year: 1947, stocks: -0.068927404, bonds: -0.086839394 },
      { year: 1948, stocks: 0.081985049, bonds: 0.022911089 },
      { year: 1949, stocks: 0.201119235, bonds: 0.044244155 },
      { year: 1950, stocks: 0.245138388, bonds: -0.072616041 },
      { year: 1951, stocks: 0.168262728, bonds: -0.025465871 },
      { year: 1952, stocks: 0.142581243, bonds: 0.010770555 },
      { year: 1953, stocks: 0.018769563, bonds: 0.048427859 },
      { year: 1954, stocks: 0.479252401, bonds: 0.020209459 },
      { year: 1955, stocks: 0.284589696, bonds: -0.000760972 },
      { year: 1956, stocks: 0.037794612, bonds: -0.044388706 },
      { year: 1957, stocks: -0.09094896, bonds: 0.031869181 },
      { year: 1958, stocks: 0.384396155, bonds: -0.056945231 },
      { year: 1959, stocks: 0.065453765, bonds: -0.023057241 },
      { year: 1960, stocks: 0.047577968, bonds: 0.099212854 },
      { year: 1961, stocks: 0.183027497, bonds: 0.012440915 },
      { year: 1962, stocks: -0.038619377, bonds: 0.047597618 },
      { year: 1963, stocks: 0.192707142, bonds: -0.004086925 },
      { year: 1964, stocks: 0.148861779, bonds: 0.0309713 },
      { year: 1965, stocks: 0.095166489, bonds: -0.009896302 },
      { year: 1966, stocks: -0.095315983, bonds: 0.01716788 },
      { year: 1967, stocks: 0.120357272, bonds: -0.057501399 },
      { year: 1968, stocks: 0.059684363, bonds: -0.025002896 },
      { year: 1969, stocks: -0.138693862, bonds: -0.111991903 },
      { year: 1970, stocks: 0.02136496, bonds: 0.139229001 },
      { year: 1971, stocks: 0.103888403, bonds: 0.051035271 },
      { year: 1972, stocks: 0.137211976, bonds: -0.011585366 },
      { year: 1973, stocks: -0.234589179, bonds: -0.058263121 },
      { year: 1974, stocks: -0.293907547, bonds: -0.069841875 },
      { year: 1975, stocks: 0.304365083, bonds: -0.002633017 },
      { year: 1976, stocks: 0.057316196, bonds: 0.06326751 },
      { year: 1977, stocks: -0.148171389, bonds: -0.04325264 },
      { year: 1978, stocks: 0.064012608, bonds: -0.077578773 },
      { year: 1979, stocks: 0.02862788, bonds: -0.133487064 },
      { year: 1980, stocks: 0.127335413, bonds: -0.099705237 },
      { year: 1981, stocks: -0.143773261, bonds: -0.052013844 },
      { year: 1982, stocks: 0.254772423, bonds: 0.38072524 },
      { year: 1983, stocks: 0.155427651, bonds: -0.003105255 },
      { year: 1984, stocks: 0.042575294, bonds: 0.110719734 },
      { year: 1985, stocks: 0.216777561, bonds: 0.223042901 },
      { year: 1986, stocks: 0.295221877, bonds: 0.224918018 },
      { year: 1987, stocks: -0.061736958, bonds: -0.064219429 },
      { year: 1988, stocks: 0.126998946, bonds: 0.014614149 },
      { year: 1989, stocks: 0.16927794, bonds: 0.095683788 },
      { year: 1990, stocks: -0.061325627, bonds: 0.038607007 },
      { year: 1991, stocks: 0.286116762, bonds: 0.133778296 },
      { year: 1992, stocks: 0.043425102, bonds: 0.07038691 },
      { year: 1993, stocks: 0.089591769, bonds: 0.100412149 },
      { year: 1994, stocks: -0.015970588, bonds: -0.09836676 },
      { year: 1995, stocks: 0.317357576, bonds: 0.210483105 },
      { year: 1996, stocks: 0.236169409, bonds: -0.034649731 },
      { year: 1997, stocks: 0.259378702, bonds: 0.132350779 },
      { year: 1998, stocks: 0.293540447, bonds: 0.103673861 },
      { year: 1999, stocks: 0.124976579, bonds: -0.111056101 },
      { year: 2000, stocks: -0.086240521, bonds: 0.143935586 },
      { year: 2001, stocks: -0.144505386, bonds: 0.048433436 },
      { year: 2002, stocks: -0.221476529, bonds: 0.103160274 },
      { year: 2003, stocks: 0.261541294, bonds: 0.011524272 },
      { year: 2004, stocks: 0.030013182, bonds: 0.00690476 },
      { year: 2005, stocks: 0.059203082, bonds: -0.012677039 },
      { year: 2006, stocks: 0.110891212, bonds: -8.92818e-6 },
      { year: 2007, stocks: -0.054698043, bonds: 0.089407593 },
      { year: 2008, stocks: -0.356495619, bonds: 0.147368371 },
      { year: 2009, stocks: 0.298481272, bonds: -0.09257338 },
      { year: 2010, stocks: 0.145187497, bonds: 0.043936627 },
      { year: 2011, stocks: 0.004690887, bonds: 0.128499318 },
      { year: 2012, stocks: 0.144014033, bonds: 0.007162894 },
      { year: 2013, stocks: 0.236561574, bonds: -0.073723951 },
      { year: 2014, stocks: 0.135804717, bonds: 0.118724101 },
      { year: 2015, stocks: -0.04749668, bonds: -0.011482725 },
      { year: 2016, stocks: 0.181588352, bonds: -0.036604118 },
      { year: 2017, stocks: 0.224585545, bonds: -0.010542083 },
      { year: 2018, stocks: -0.062028067, bonds: 0.001978359 },
      { year: 2019, stocks: 0.250408607, bonds: 0.083400178 },
      { year: 2020, stocks: 0.162378785, bonds: 0.05827981 },
    ]

    const _logReturns = (returns: number[]) =>
      returns.map((x) => Math.log(1 + x))

    const _stats = (returns: number[]) => {
      const expectedValue = _.mean(returns)
      const variance =
        _.sumBy(returns, (x) => Math.pow(x - expectedValue, 2)) /
        (returns.length - 1)
      const standardDeviation = Math.sqrt(variance)

      return { returns, expectedValue, variance, standardDeviation }
    }

    // Empirically determined to be more accurate.
    const deltaCalculated = { stocks: 0.0006162, bonds: 0.0000005 }

    const statsFn = (
      returns: number[],
      delta: 'stocks' | 'bonds' | number = 0,
    ) => {
      const log = _stats(_logReturns(returns))

      const deltaNum =
        typeof delta === 'number' ? delta : deltaCalculated[delta]
      const convertExpectedToLog = (x: number) =>
        Math.log(1 + x) - log.variance / 2 + deltaNum

      const adjust = (targetExpectedValue: number) => {
        const targetLogExpectedValue = convertExpectedToLog(targetExpectedValue)
        const adjustmentLogExpected = log.expectedValue - targetLogExpectedValue
        const adjustedLogReturns = log.returns.map(
          (log) => log - adjustmentLogExpected,
        )
        return adjustedLogReturns.map((x) => Math.exp(x) - 1)
      }
      return {
        ..._stats(returns),
        log,
        convertExpectedToLog,
        adjust,
      }
    }

    const historicalReturns = {
      raw,
      stocks: statsFn(
        raw.map((x) => x.stocks),
        'stocks',
      ),
      bonds: statsFn(
        raw.map((x) => x.bonds),
        'bonds',
      ),
      statsFn,
    }
  }
}
