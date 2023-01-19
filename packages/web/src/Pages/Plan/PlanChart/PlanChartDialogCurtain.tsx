import { noCase, PlanParams } from '@tpaw/common'
import React from 'react'
import { NoDisplayOnOpacity0Transition } from '../../../Utils/NoDisplayOnOpacity0Transition'
import { useSimulation } from '../../App/WithSimulation'

const _state = (dialogPosition: PlanParams['dialogPosition']) => {
  switch (dialogPosition) {
    case 'age':
    case 'current-portfolio-balance':
    case 'future-savings':
    case 'income-during-retirement':
      return 'waitingForInputs' as const
    case 'show-results':
      return 'readyToReveal' as const
    case 'show-all-inputs':
    case 'done':
      return 'hide' as const
    default:
      noCase(dialogPosition)
  }
}
export const PlanChartDialogCurtain = React.memo(
  ({ layout }: { layout: 'mobile' | 'laptop' | 'desktop' }) => {
    const { params } = useSimulation()
    const state = _state(params.dialogPosition)

    return (
      <NoDisplayOnOpacity0Transition
        className="absolute inset-0  z-10 bg-chartBG  flex justify-center items-center "
        style={{
          transitionProperty: 'opacity',
          transitionDuration: '1000ms',
          opacity: `${state === 'hide' ? 0 : 1}`,
        }}
      >
        {state === 'waitingForInputs' ? (
          <div className="">
            <div className="flex justify-center">
              <div className="border border-dashed border-gray-700 px-5 py-2 rounded-xl ">
                <h2 className="text-xl font-medium lighten">
                  Waiting for your inputs...
                </h2>
              </div>
            </div>
            <div className={`max-w-[600px] mt-5 sm:mt-10`}>
              <p className="p-base mt-2 px-4">
                {layout === 'mobile'
                  ? `This panel displays the results of simulating your retirement
                based on your current inputs. Age and Wealth are the minimum
                inputs needed to simulate your retirement. After you complete
                those sections, this panel will become active.`
                  : `This panel displays the results of simulating your retirement
                based on your current inputs. Age and Wealth are the minimum
                inputs needed to simulate your retirement. After you complete
                those sections, this panel will become active.`}
              </p>
            </div>
          </div>
        ) : state === 'readyToReveal' ? (
          <div className="">
            <div className="flex justify-center">
              <button
                id="planChartDialogCurtionShowResultsButton"
                className="text-xl px-8 py-3 btn-dark "
              >
                Show Results Panel
              </button>
            </div>
          </div>
        ) : state === 'hide' ? (
          <></>
        ) : (
          noCase(state)
        )}
      </NoDisplayOnOpacity0Transition>
    )
  },
)
