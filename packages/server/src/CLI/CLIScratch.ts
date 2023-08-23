import jsonpath from 'fast-json-patch'
import { performance } from 'perf_hooks'
import { cli } from './CLI.js'
import { cloneJSON } from '../Utils/CloneJSON.js'

cli.command('scratch ').action(async () => {
  const diff = jsonpath.compare(left, right)

  const start = performance.now()
  const n = 100000
  for (let i = 0; i < n; i++) {
    jsonpath.applyPatch(cloneJSON(left), cloneJSON(diff), false).newDocument
  }
  const end = performance.now()
  console.log(end - start)

  // console.dir(jsonpath.applyPatch(left, diff, false).newDocument === left)
  // console.dir(left)
})

const left = {
  v: 21,
  risk: {
    swr: {
      withdrawal: {
        type: 'default',
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
        type: 'retirementDateSpecified',
        maxAge: {
          inMonths: 1200,
        },
        monthOfBirth: {
          year: 1988,
          month: 5,
        },
        retirementAge: {
          inMonths: 780,
        },
      },
    },
    withPartner: false,
  },
  wealth: {
    futureSavings: [
      {
        id: 0,
        label: 'From My Salary',
        value: 300,
        nominal: false,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            type: 'calendarMonthAsNow',
            monthOfEntry: {
              year: 2023,
              month: 5,
            },
          },
        },
      },
      {
        id: 1,
        label: 'Firs sourxc',
        value: 0,
        nominal: false,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            type: 'calendarMonthAsNow',
            monthOfEntry: {
              year: 2024,
              month: 6,
            },
          },
        },
      },
    ],
    portfolioBalance: {
      amount: 7171,
      timestamp: 1716840813275,
    },
    incomeDuringRetirement: [
      {
        id: 0,
        label: 'SOme long text',
        value: 500,
        nominal: false,
        monthRange: {
          end: {
            age: 'max',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            age: 'retirement',
            type: 'namedAge',
            person: 'person1',
          },
        },
      },
      {
        id: 1,
        label: 'SEcond sournce',
        value: 200,
        nominal: false,
        monthRange: {
          end: {
            age: 'max',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            age: 'retirement',
            type: 'namedAge',
            person: 'person1',
          },
        },
      },
      {
        id: 2,
        label: 'Third source',
        value: 0,
        nominal: false,
        monthRange: {
          end: {
            age: 'max',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            age: 'retirement',
            type: 'namedAge',
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
        type: 'suggested',
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
    },
  },
  timestamp: 1718309793892,
  dialogPosition: 'done',
  adjustmentsToSpending: {
    tpawAndSPAW: {
      legacy: {
        total: 0,
        external: [],
      },
      monthlySpendingFloor: 1000,
      monthlySpendingCeiling: 7000,
    },
    extraSpending: {
      essential: [
        {
          id: 0,
          label: 'MOre long test',
          value: 0,
          nominal: false,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
            numMonths: 60,
          },
        },
      ],
      discretionary: [
        {
          id: 0,
          label: 'Spending another thingd',
          value: 100,
          nominal: false,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
            numMonths: 60,
          },
        },
      ],
    },
  },
}

const right = {
  v: 21,
  risk: {
    swr: {
      withdrawal: {
        type: 'default',
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
        type: 'retirementDateSpecified',
        maxAge: {
          inMonths: 1200,
        },
        monthOfBirth: {
          year: 1988,
          month: 5,
        },
        retirementAge: {
          inMonths: 780,
        },
      },
    },
    withPartner: false,
  },
  wealth: {
    futureSavings: [
      {
        id: 0,
        label: 'From My Salary',
        value: 300,
        nominal: false,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            type: 'calendarMonthAsNow',
            monthOfEntry: {
              year: 2023,
              month: 5,
            },
          },
        },
      },
      {
        id: 1,
        label: 'Firs sourxc',
        value: 0,
        nominal: false,
        monthRange: {
          end: {
            age: 'lastWorkingMonth',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            type: 'calendarMonthAsNow',
            monthOfEntry: {
              year: 2024,
              month: 6,
            },
          },
        },
      },
    ],
    portfolioBalance: {
      amount: 7171,
      timestamp: 17167940813275,
    },
    incomeDuringRetirement: [
      {
        id: 0,
        label: 'SOme long text',
        value: 500,
        nominal: false,
        monthRange: {
          end: {
            age: 'max',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            age: 'retirement',
            type: 'namedAge',
            person: 'person1',
          },
        },
      },
      {
        id: 1,
        label: 'SEcond sournce',
        value: 200,
        nominal: false,
        monthRange: {
          end: {
            age: 'max',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            age: 'retirement',
            type: 'namedAge',
            person: 'person1',
          },
        },
      },
      {
        id: 2,
        label: 'Third source',
        value: 0,
        nominal: false,
        monthRange: {
          end: {
            age: 'max',
            type: 'namedAge',
            person: 'person1',
          },
          type: 'startAndEnd',
          start: {
            age: 'retirement',
            type: 'namedAge',
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
        type: 'suggested',
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
    },
  },
  timestamp: 1718309793894,
  dialogPosition: 'done',
  adjustmentsToSpending: {
    tpawAndSPAW: {
      legacy: {
        total: 0,
        external: [],
      },
      monthlySpendingFloor: 1000,
      monthlySpendingCeiling: 7000,
    },
    extraSpending: {
      essential: [
        {
          id: 0,
          label: 'MOre long test',
          value: 0,
          nominal: false,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
            numMonths: 60,
          },
        },
      ],
      discretionary: [
        {
          id: 0,
          label: 'Spending another thingd',
          value: 100,
          nominal: false,
          monthRange: {
            type: 'startAndNumMonths',
            start: {
              age: 'retirement',
              type: 'namedAge',
              person: 'person1',
            },
            numMonths: 60,
          },
        },
      ],
    },
  },
}
