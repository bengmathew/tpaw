import {Document} from '@contentful/rich-text-types'
import {Transition} from '@headlessui/react'
import React, {Dispatch} from 'react'
import ReactDOM from 'react-dom'
import {Contentful} from '../../../Utils/Contentful'
import {noCase} from '../../../Utils/Utils'
import {useSimulation} from '../../App/WithSimulation'
import {usePlanContent} from '../Plan'
import {
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
} from './ChartPanelType'

export const ChartPanelDescription = React.memo(
  ({
    className = '',
    type,
    showDescriptionPopUp,
    setShowDescriptionPopUp,
    ref,
  }: {
    className?: string
    type: ChartPanelType | 'sharpe-ratio'
    showDescriptionPopUp: boolean
    setShowDescriptionPopUp: Dispatch<boolean>
    ref?: (x: HTMLElement | null) => void
  }) => {
    const {tpawResult, params} = useSimulation()
    const {numRuns} = tpawResult
    const content = useContent(type)

    return (
      <div className={`${className} text-lg sm:text-lg font-font2`} ref={ref}>
        <div className="hidden sm:block">
          <Contentful.RichText
            body={Contentful.replaceVariables(
              {numRuns: `${numRuns}`},
              content.intro[params.strategy]
            )}
            p="text-base inline lighten"
          />{' '}
          <button
            className="bg-gray-300 text-gray-500 px-2 py-0 rounded-full text-base"
            onClick={() => setShowDescriptionPopUp(true)}
          >
            Details
          </button>
        </div>
        {ReactDOM.createPortal(
          <Transition
            className="page fixed inset-0 flex flex-col justify-center items-center z-0"
            show={showDescriptionPopUp}
            onClick={() => setShowDescriptionPopUp(false)}
          >
            <Transition.Child
              className="fixed inset-0 bg-black bg-opacity-70 z-0 "
              enter="transition-opacity duration-300"
              enterFrom="opacity-0"
              leave="transition-opacity duration-300"
              leaveTo="opacity-0"
            />

            <Transition.Child
              className="flex flex-col w-[600px] max-w-[100vw] rounded-xl bg-pageBG  max-h-[calc(100vh-50px)] overflow-scroll z-10 p-2 sm:p-4"
              enter="transition-transfrom duration-300"
              enterFrom="scale-95  opacity-0"
              leave="transition-transfrom duration-300"
              leaveTo="scale-95 opacity-0"
              style={{boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)'}}
            >
              <_RichText className="">
                {Contentful.replaceVariables(
                  {numRuns: `${numRuns}`},
                  content.body[params.strategy]
                )}
              </_RichText>
            </Transition.Child>
          </Transition>,
          window.document.body
        )}
      </div>
    )
  }
)

function useContent(type: ChartPanelType | 'sharpe-ratio') {
  const content = usePlanContent().chart
  switch (type) {
    case 'spending-total':
      return content.spending.total
    case 'spending-general':
      return content.spending.regular
    case 'portfolio':
      return content.portfolio
    case 'glide-path':
      return content.glidePath
    case 'withdrawal':
      return content.withdrawalRate
    case 'sharpe-ratio':
      return content.sharpeRatio
    default:
      if (isChartPanelSpendingEssentialType(type))
        return content.spending.essential
      if (isChartPanelSpendingDiscretionaryType(type))
        return content.spending.discretionary
      noCase(type)
  }
}

const _RichText = React.memo(
  ({className = '', children}: {className?: string; children: Document}) => {
    return (
      <div className={`${className}`}>
        <Contentful.RichText
          body={children}
          ul="list-disc ml-5"
          ol="list-decimal ml-5"
          p="p-base mb-3"
          h1="font-bold text-lg mb-3"
          h2="font-bold text-lg mt-6 mb-3"
          a="underline"
          aExternalLink="text-[12px] ml-1"
        />
      </div>
    )
  }
)
