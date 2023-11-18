import {
  API,
  SomeNonPlanParams,
  defaultNonPlanParams,
  nonPlanParamsBackwardsCompatibleGuard,
  nonPlanParamsMigrate,
} from '@tpaw/common'
import clsx from 'clsx'
import _ from 'lodash'
import React from 'react'
import { paddingCSS } from '../../../../../Utils/Geometry'
import { NumberInput } from '../../../../Common/Inputs/NumberInput'
import { smartDeltaFnForMonthlyAmountInput } from '../../../../Common/Inputs/SmartDeltaFnForAmountInput'
import { ToggleSwitch } from '../../../../Common/Inputs/ToggleSwitch'
import { useNonPlanParams } from '../../../PlanRootHelpers/WithNonPlanParams'
import { useGetPlanResultsChartURL } from '../../PlanResults/UseGetPlanResultsChartURL'
import { usePlanResultsChartType } from '../../PlanResults/UsePlanResultsChartType'
import { useChartData } from '../../WithPlanResultsChartData'
import { PlanInputModifiedBadge } from '../Helpers/PlanInputModifiedBadge'
import {
  PlanInputBody,
  PlanInputBodyPassThruProps,
} from '../PlanInputBody/PlanInputBody'
import { chain, json, string } from 'json-guard'

export const PlanInputDevMisc = React.memo(
  (props: PlanInputBodyPassThruProps) => {
    return (
      <PlanInputBody {...props}>
        <div className="">
          <_MiscCard className="mt-10" props={props} />
        </div>
      </PlanInputBody>
    )
  },
)

const _MiscCard = React.memo(
  ({
    className = '',
    props,
  }: {
    className?: string
    props: PlanInputBodyPassThruProps
  }) => {
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const getPlanChartURL = useGetPlanResultsChartURL()

    const isModified = useIsPlanInputDevMiscModified()

    const handleChangeShowAllMonths = (x: boolean) => {
      const clone = _.cloneDeep(nonPlanParams)
      clone.dev.alwaysShowAllMonths = x
      setNonPlanParams(clone)
    }

    return (
      <div
        className={`${className} params-card relative`}
        style={{ padding: paddingCSS(props.sizing.cardPadding) }}
      >
        <PlanInputModifiedBadge show={isModified} mainPage={false} />
        <div className=" flex justify-start gap-x-4 items-center mt-4">
          <ToggleSwitch
            className=""
            checked={nonPlanParams.dev.alwaysShowAllMonths}
            setChecked={(x) => handleChangeShowAllMonths(x)}
          />
          <h2 className=""> Show All Months</h2>
        </div>
        <_ChartYRangeOverride className="mt-4" />

        <button
          className="block btn-sm btn-outline mt-4"
          onClick={() => {
            console.dir('what')
            console.dir(API.version)
            const check = chain(
              string,
              json,
              nonPlanParamsBackwardsCompatibleGuard,
            )(JSON.stringify(testNPP))
            console.dir(check)
          }}
        >
          Test
        </button>
        <button
          className="block btn-sm btn-outline mt-4"
          onClick={() => {
            throw new Error('Crash Test')
          }}
        >
          Crash
        </button>
        <button
          className="mt-6 underline disabled:lighten-2 block"
          onClick={() =>
            handleChangeShowAllMonths(
              defaultNonPlanParams.dev.alwaysShowAllMonths,
            )
          }
          disabled={!isModified}
        >
          Reset to Default
        </button>
      </div>
    )
  },
)

const _ChartYRangeOverride = React.memo(
  ({ className }: { className?: string }) => {
    const { nonPlanParams, setNonPlanParams } = useNonPlanParams()
    const chartType = usePlanResultsChartType()
    const chartData = useChartData(chartType)
    return (
      <div className={clsx(className)}>
        <div className="flex justify-start gap-x-4 items-center ">
          <ToggleSwitch
            checked={!!nonPlanParams.dev.overridePlanResultChartYRange}
            setChecked={(checked) => {
              const clone = _.cloneDeep(nonPlanParams)
              if (!checked) {
                clone.dev.overridePlanResultChartYRange = false
              } else {
                clone.dev.overridePlanResultChartYRange =
                  chartData.displayRange.y
              }
              setNonPlanParams(clone)
            }}
          />
          <h2 className=""> Override Y Range</h2>
        </div>
        {!!nonPlanParams.dev.overridePlanResultChartYRange && (
          <div
            className="ml-[50px] mt-3 inline-grid gap-x-4 items-center"
            style={{ grid: 'auto/auto 100px' }}
          >
            <h2 className="">Max Y</h2>
            <NumberInput
              value={nonPlanParams.dev.overridePlanResultChartYRange.end}
              textAlign="right"
              width={125}
              setValue={(end) => {
                if (end <= 0) return true
                const clone = _.cloneDeep(nonPlanParams)
                clone.dev.overridePlanResultChartYRange = { start: 0, end }
                setNonPlanParams(clone)
                return false
              }}
              modalLabel={'Max Y'}
              increment={smartDeltaFnForMonthlyAmountInput.increment}
              decrement={smartDeltaFnForMonthlyAmountInput.decrement}
            />
          </div>
        )}
      </div>
    )
  },
)

export const useIsPlanInputDevMiscModified = () => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    nonPlanParams.dev.alwaysShowAllMonths !==
      defaultNonPlanParams.dev.alwaysShowAllMonths ||
    nonPlanParams.dev.overridePlanResultChartYRange !==
      defaultNonPlanParams.dev.overridePlanResultChartYRange
  )
}

export const PlanInputDevMiscSummary = React.memo(() => {
  const { nonPlanParams } = useNonPlanParams()
  return (
    <>
      <h2>
        Always show all months:{' '}
        {nonPlanParams.dev.alwaysShowAllMonths ? 'yes' : 'no'}
      </h2>
      <h2 className="">
        Override Y Range:{' '}
        {nonPlanParams.dev.overridePlanResultChartYRange
          ? `${new Intl.NumberFormat('en-US', {
              minimumFractionDigits: 0,
              maximumFractionDigits: 0,
            }).format(nonPlanParams.dev.overridePlanResultChartYRange.end)}`
          : 'no'}
      </h2>
    </>
  )
})

// TODO: Delete
const testNPP: SomeNonPlanParams = {
  v: 20,
  plan: {
    risk: {
      swr: {
        withdrawal: {
          type: 'asPercentPerYear',
          percentPerYear: 0.029,
        },
      },
      spaw: {
        annualSpendingTilt: 0.008,
      },
      tpaw: {
        riskTolerance: {
          at20: 12,
          deltaAtMaxAge: -2,
          forLegacyAsDeltaFromAt20: 2,
        },
        timePreference: 0,
        additionalAnnualSpendingTilt: 0,
      },
      spawAndSWR: {
        allocation: {
          end: {
            stocks: 0.5,
          },
          start: {
            month: {
              year: 2023,
              month: 5,
            },
            stocks: 0.5,
          },
          intermediate: [],
        },
      },
      tpawAndSPAW: {
        lmp: 0,
      },
    },
    people: {
      person1: {
        ages: {
          type: 'retiredWithNoRetirementDateSpecified',
          maxAge: {
            inMonths: 1200,
          },
          monthOfBirth: {
            year: 1963,
            month: 7,
          },
        },
      },
      person2: {
        ages: {
          type: 'retiredWithNoRetirementDateSpecified',
          maxAge: {
            inMonths: 1200,
          },
          monthOfBirth: {
            year: 1965,
            month: 6,
          },
        },
      },
      withPartner: true,
      withdrawalStart: 'person1',
    },
    wealth: {
      futureSavings: [],
      portfolioBalance: {
        history: {
          monthBoundaryDetails: [
            {
              allocation: {
                stocks: 0.5927,
              },
              startOfMonth: {
                year: 2023,
                month: 7,
              },
              netContributionOrWithdrawal: {
                type: 'withdrawal',
                withdrawal: 1200,
              },
            },
          ],
          planChangeStockAllocations: [
            {
              allocation: {
                stocks: 0.6519,
              },
              effectiveAtMarketCloseTime: 1685649600000,
            },
            {
              allocation: {
                stocks: 0.614,
              },
              effectiveAtMarketCloseTime: 1685736000000,
            },
            {
              allocation: {
                stocks: 0.6092,
              },
              effectiveAtMarketCloseTime: 1686600000000,
            },
            {
              allocation: {
                stocks: 0.5,
              },
              effectiveAtMarketCloseTime: 1686686400000,
            },
            {
              allocation: {
                stocks: 0.6283,
              },
              effectiveAtMarketCloseTime: 1687291200000,
            },
            {
              allocation: {
                stocks: 0.5324,
              },
              effectiveAtMarketCloseTime: 1690488000000,
            },
          ],
        },
        original: {
          amount: 400000,
          timestamp: 1685729090725,
        },
        isLastPlanChange: false,
      },
      retirementIncome: [
        {
          id: 0,
          label: 'Social Security',
          value: 3000,
          nominal: false,
          monthRange: {
            end: {
              age: 'max',
              type: 'namedAge',
              person: 'person1',
            },
            type: 'startAndEnd',
            start: {
              age: {
                inMonths: 840,
              },
              type: 'numericAge',
              person: 'person1',
            },
          },
        },
      ],
    },
    advanced: {
      sampling: 'monteCarlo',
      strategy: 'TPAW',
      annualReturns: {
        expected: {
          type: 'manual',
          bonds: 0.046,
          stocks: 0.039,
        },
        historical: {
          type: 'adjusted',
          adjustment: {
            type: 'toExpected',
          },
          correctForBlockSampling: true,
        },
      },
      annualInflation: {
        type: 'suggested',
      },
      monteCarloSampling: {
        blockSize: 60,
        numOfSimulations: 500,
      },
    },
    timestamp: 1690493345826,
    dialogPosition: 'done',
    adjustmentsToSpending: {
      tpawAndSPAW: {
        legacy: {
          total: 100000,
          external: [],
        },
        monthlySpendingFloor: null,
        monthlySpendingCeiling: 1000,
      },
      extraSpending: {
        essential: [
          {
            id: 0,
            label: null,
            value: 100,
            nominal: false,
            monthRange: {
              type: 'startAndNumMonths',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 6,
                },
              },
              numMonths: 60,
            },
          },
        ],
        discretionary: [
          {
            id: 0,
            label: null,
            value: 100,
            nominal: false,
            monthRange: {
              type: 'startAndNumMonths',
              start: {
                type: 'calendarMonthAsNow',
                monthOfEntry: {
                  year: 2023,
                  month: 6,
                },
              },
              numMonths: 60,
            },
          },
        ],
      },
    },
  },
  nonPlan: {
    dev: {
      alwaysShowAllMonths: false,
      currentTimeFastForward: {
        shouldFastForward: false,
      },
    },
    defaultTimezone: {
      type: 'auto',
      ianaTimezoneName: 'America/Los_Angeles',
    },
    percentileRange: {
      end: 95,
      start: 5,
    },
    migrationWarnings: {
      v14tov15: true,
      v16tov17: true,
      v19tov20: true,
    },
  },
}
