import {faChevronRight} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import {gsap} from 'gsap'
import _ from 'lodash'
import React, {
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react'
import ReactDOM from 'react-dom'
import {Contentful} from '../../../Utils/Contentful'
import {Padding, RectExt, rectExt} from '../../../Utils/Geometry'
import {fGet, noCase} from '../../../Utils/Utils'
import {useWindowSize} from '../../../Utils/WithWindowSize'
import {useSimulation} from '../../App/WithSimulation'
import {chartDrawDataLines} from '../../Common/Chart/ChartComponent/ChartDrawDataLines'
import {
  ChartReact,
  ChartReactState,
  ChartReactStatefull,
} from '../../Common/Chart/ChartReact'
import {ChartUtils} from '../../Common/Chart/ChartUtils/ChartUtils'
import {usePlanContent} from '../Plan'
import {chartPanelLabel} from './ChartPanelLabel'
import {ChartPanelMenuButton} from './ChartPanelMenuButton'
import {
  chartPanelSpendingDiscretionaryTypeID,
  chartPanelSpendingEssentialTypeID,
  ChartPanelType,
  isChartPanelSpendingDiscretionaryType,
  isChartPanelSpendingEssentialType,
} from './ChartPanelType'
import {
  tpawChartDataMain,
  TPAWChartDataMain,
} from './TPAWChart/TPAWChartDataMain'

const duration = 0.5
const scale = 0.95
const widthForFullSize = 700
export type ChartPanelMenuStateful = {
  setButtonScale: (scale: number) => void
}
type Props = {
  className?: string
  type: ChartPanelType
  onSelect: (type: ChartPanelType) => void
  layout: 'mobile' | 'desktop' | 'laptop'
  showDescriptionPopUp: () => void
}
export const ChartPanelMenu = React.memo(
  React.forwardRef<ChartPanelMenuStateful, Props>(
    (
      {className = '', type, onSelect, layout, showDescriptionPopUp}: Props,
      forwardRef
    ) => {
      const windowSize = useWindowSize()
      const simulation = useSimulation()
      const {params} = simulation.tpawResult.args

      const [referenceElement, setReferenceElement] =
        useState<HTMLDivElement | null>(null)

      useImperativeHandle(
        forwardRef,
        () => ({
          setButtonScale: scale => {
            if (!referenceElement) return
            referenceElement.style.transform = `scale(${scale})`
          },
        }),
        [referenceElement]
      )

      const [portalElement, setPortalElement] = useState<HTMLDivElement | null>(
        null
      )
      const [overlayElement, setOverlayElement] =
        useState<HTMLDivElement | null>(null)
      const [popperElement, setPopperElement] = useState<HTMLDivElement | null>(
        null
      )
      const allChartData = useAllChartData()
      // Hack to force redraw on open. Seemed like the draws were not taking
      // effect when the canvas was not visible.
      const [drawKey, setDrawKey] = useState(0)

      const handleShow = () => {
        setDrawKey(x => x + 1)
        fGet(portalElement).style.visibility = 'visible'
        fGet(portalElement).style.pointerEvents = 'auto'
        fGet(overlayElement).style.opacity = '.7'
        fGet(popperElement).style.opacity = '1'

        const position = fGet(referenceElement).getBoundingClientRect()
        const menuWidth =
          fGet(popperElement).getBoundingClientRect().width / scale // .95 to account for scaling.
        fGet(popperElement).style.top = `${position.top}px`
        fGet(popperElement).style.left =
          windowSize.width < widthForFullSize
            ? `0px`
            : `${Math.min(position.left, windowSize.width - menuWidth - 20)}px`
        const timeline = gsap.timeline()
        timeline.fromTo(popperElement, {scale: 0.95}, {scale: 1, duration}, 0)
      }
      const handleHide = (type: ChartPanelType | null) => {
        if (type) onSelect(type)
        fGet(overlayElement).style.opacity = '0'
        fGet(popperElement).style.opacity = '0'
        const timeline = gsap.timeline({
          onComplete: () => {
            fGet(portalElement).style.visibility = 'hidden'
            fGet(portalElement).style.pointerEvents = 'none'
          },
        })
        timeline.to(popperElement, {scale, duration}, 0)
      }

      const buttonProps = {
        currType: type,
        onSelect: (x: ChartPanelType) => handleHide(x),
        allChartData,
        drawKey,
      }

      return (
        <>
          <ChartPanelMenuButton
            layout={layout}
            showDescriptionPopUp={showDescriptionPopUp}
            className={className}
            type={type}
            onClick={handleShow}
            ref={x => setReferenceElement(x)}
          />

          {ReactDOM.createPortal(
            <div
              className=" page fixed inset-0"
              ref={setPortalElement}
              style={{visibility: 'hidden', pointerEvents: 'none'}}
            >
              <div
                className="fixed inset-0 bg-black"
                ref={setOverlayElement}
                onClick={() => handleHide(null)}
                style={{
                  opacity: 0,
                  transitionProperty: 'opacity',
                  transitionDuration: `${duration}s`,
                }}
              />
              <div
                className={`flex absolute flex-col  rounded-xl  bg-pageBG  max-h-[calc(100vh-50px)] overflow-scroll`}
                ref={setPopperElement}
                style={{
                  opacity: 0,
                  transitionProperty: 'opacity',
                  transitionDuration: `${duration}s`,
                  width:
                    windowSize.width < widthForFullSize
                      ? `${windowSize.width}px`
                      : `${widthForFullSize}px`,
                  boxShadow: '0px 0px 10px 5px rgba(0,0,0,0.28)',
                }}
              >
                <_Button type="spending-total" {...buttonProps} />
                {(params.withdrawals.fundedByBonds.length > 0 ||
                  params.withdrawals.fundedByRiskPortfolio.length > 0) && (
                  <div className=" flex flex-col border- border-gray-400 ">
                    <_Button
                      className="pl-10"
                      type="spending-general"
                      {...buttonProps}
                    />
                    {params.withdrawals.fundedByBonds.map(x => (
                      <_Button
                        className="pl-10"
                        key={`essential-${x.id}`}
                        type={`spending-essential-${x.id}`}
                        {...buttonProps}
                      />
                    ))}
                    {params.withdrawals.fundedByRiskPortfolio.map(x => (
                      <_Button
                        className="pl-10"
                        key={`discretionary-${x.id}`}
                        type={`spending-discretionary-${x.id}`}
                        {...buttonProps}
                      />
                    ))}
                  </div>
                )}
                <_Button type={'portfolio'} {...buttonProps} />
                <_Button type={'glide-path'} {...buttonProps} />
                <_Button type={'withdrawal-rate'} {...buttonProps} />
              </div>
            </div>,
            window.document.body
          )}
        </>
      )
    }
  )
)

const _Button = React.memo(
  ({
    className = '',
    currType,
    onSelect,
    type,
    allChartData,
    drawKey,
  }: {
    className?: string
    currType: ChartPanelType
    onSelect: (type: ChartPanelType) => void
    type: ChartPanelType
    allChartData: _AllChartData
    drawKey: number
  }) => {
    const {params} = useSimulation().tpawResult.args
    const [description, chartData] = useInfo(type, allChartData)
    const {label, subLabel} = chartPanelLabel(params, type, 'short')
    const isCurrent = _.isEqual(currType, type)
    const windowSize = useWindowSize()
    const width = windowSize.width < 640 ? 120 : 145
    const height = windowSize.width < 640 ? 50 : 55
    return (
      <button
        className={`${className} text-left px-4 py-2  grid  gap-x-4 items-center
        ${isCurrent ? 'bg-gray-200 ' : ''} `}
        onClick={() => onSelect(type)}
        style={{grid: 'auto / 1fr auto'}}
      >
        <div className="">
          <h2 className="text-base sm:text-lg font-bold">
            {label.map((x, i) => (
              <React.Fragment key={i}>
                <span>{x}</span>
                {i !== label.length - 1 && (
                  <FontAwesomeIcon
                    className="mx-2 text-xs sm:text-sm lighten-2"
                    icon={faChevronRight}
                  />
                )}
              </React.Fragment>
            ))}
          </h2>
          {subLabel && (
            <h2 className="text-sm sm:text-base font-semibold -mt-1">
              {subLabel}
            </h2>
          )}
          <Contentful.RichText
            body={description.fields.body}
            p={`${isCurrent ? '' : 'lighten-2'} text-sm -mt-1`}
          />
        </div>
        <div
          className={`relative  border rounded-xl  flex flex-col justify-center 
          ${
            isCurrent
              ? 'bg-gray-300 border-gray-300 '
              : 'bg-gray-200 border-gray-300 '
          }`}
          style={{width: `${width}px`, height: `${height}px`}}
        >
          <_Chart
            data={chartData}
            isCurrent={isCurrent}
            drawKey={drawKey}
            startingSizing={{
              position: rectExt({x: 0, y: 0, width, height}),
              padding: {left: 10, right: 10, top: 5, bottom: 5},
            }}
          />
        </div>
      </button>
    )
  }
)

const useInfo = (panelType: ChartPanelType, allData: _AllChartData) => {
  const content = usePlanContent().chart
  switch (panelType) {
    case 'spending-total':
      return [content.spending.total.menu, allData.spending.total] as const
    case 'spending-general': {
      return [content.spending.regular.menu, allData.spending.regular] as const
    }
    case 'portfolio':
      return [content.portfolio.menu, allData.portfolio] as const
    case 'glide-path':
      return [content.glidePath.menu, allData.glidePath] as const
    case 'withdrawal-rate':
      return [content.withdrawalRate.menu, allData.withdrawalRate] as const
    default:
      if (isChartPanelSpendingEssentialType(panelType)) {
        const id = chartPanelSpendingEssentialTypeID(panelType)
        return [
          content.spending.essential.menu,
          fGet(allData.spending.essential.get(id)),
        ] as const
      }
      if (isChartPanelSpendingDiscretionaryType(panelType)) {
        const id = chartPanelSpendingDiscretionaryTypeID(panelType)
        return [
          content.spending.discretionary.menu,
          fGet(allData.spending.discretionary.get(id)),
        ] as const
      }
      noCase(panelType)
  }
}

type _AllChartData = ReturnType<typeof useAllChartData>
function useAllChartData() {
  const {tpawResult, highlightPercentiles} = useSimulation()
  return useMemo(() => {
    const {params} = tpawResult.args
    const _data = (type: ChartPanelType) =>
      tpawChartDataMain(type, tpawResult, highlightPercentiles)
    const result = {
      spending: {
        total: _data('spending-total'),
        regular: _data('spending-general'),
        essential: new Map(
          params.withdrawals.fundedByBonds.map(x => [
            x.id,
            _data(`spending-essential-${x.id}`),
          ])
        ),
        discretionary: new Map(
          params.withdrawals.fundedByRiskPortfolio.map(x => [
            x.id,
            _data(`spending-discretionary-${x.id}`),
          ])
        ),
      },
      portfolio: _data('portfolio'),
      glidePath: _data('glide-path'),
      withdrawalRate: _data('withdrawal-rate'),
    }
    return result
  }, [tpawResult, highlightPercentiles])
}

const _processData = (
  data: TPAWChartDataMain
): ChartReactState<TPAWChartDataMain> => ({
  data,
  xyRange: {
    x: data.years.displayRange,
    y: {start: data.min.y, end: data.max.y},
  },
  animation: null,
})

const _Chart = React.memo(
  ({
    data,
    isCurrent,
    drawKey,
    startingSizing,
  }: {
    data: TPAWChartDataMain
    isCurrent: boolean
    drawKey: number
    startingSizing: {position: RectExt; padding: Padding}
  }) => {
    const ref = useRef<ChartReactStatefull | null>(null)
    const [state, setState] = useState(() => _processData(data))

    useEffect(() => {
      setState(_processData(data))
    }, [data, drawKey])

    const components = useMemo(
      () => [
        chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 0.5 * 0.8,
          strokeStyle: isCurrent
            ? ChartUtils.color.gray[400]
            : ChartUtils.color.gray[400],
          dataFn: (data: TPAWChartDataMain) => ({
            lines: data.percentiles
              .filter(x => !x.isHighlighted)
              .map(x => x.data),
          }),
        }),

        chartDrawDataLines<TPAWChartDataMain>({
          lineWidth: 1.2 * 0.8,
          strokeStyle: isCurrent
            ? ChartUtils.color.gray[500]
            : ChartUtils.color.gray[500],
          dataFn: (data: TPAWChartDataMain) => ({
            lines: data.percentiles
              .filter(x => x.isHighlighted)
              .map(x => x.data),
          }),
        }),
      ],
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [isCurrent]
    )

    // return <></>
    return (
      <ChartReact<TPAWChartDataMain>
        ref={ref}
        state={state}
        starting={{sizing: startingSizing}}
        components={components}
      />
    )
  }
)
