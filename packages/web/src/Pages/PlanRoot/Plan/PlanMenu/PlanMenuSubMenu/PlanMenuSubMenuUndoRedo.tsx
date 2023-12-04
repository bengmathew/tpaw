import { faRedo, faUndo } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { PlanParams, PlanParamsChangeAction, block, fGet } from '@tpaw/common'
import clix from 'clsx'
import { formatDistance } from 'date-fns'
import _ from 'lodash'
import React, { useEffect, useMemo, useRef, useState } from 'react'
import { ContextMenu2 } from '../../../../Common/Modal/ContextMenu2'
import { processPlanParamsChangeActionCurrent } from '../../../PlanRootHelpers/PlanParamsChangeAction'
import {
  TARGET_UNDO_DEPTH,
  WorkingPlanInfo,
} from '../../../PlanRootHelpers/UseWorkingPlan'
import {
  SimulationInfoForPlanMode,
  useSimulation,
} from '../../../PlanRootHelpers/WithSimulation'
import { usePlanColors } from '../../UsePlanColors'
import { processPlanParamsChangeActionDeprecated } from '../../../PlanRootHelpers/PlanParamsChangeActionDeprecated'

export const PlanMenuSubMenuUndoRedo = React.memo(
  ({
    className,
    simulationDetailForPlanMode,
  }: {
    className?: { undo?: string; redo?: string }
    simulationDetailForPlanMode: SimulationInfoForPlanMode
  }) => {
    const planColors = usePlanColors()

    return (
      <div
        className="rounded-lg flex "
        style={{
          backgroundColor: planColors.results.bg,
          color: planColors.results.fg,
        }}
      >
        <_Menu
          className={clix(className?.undo)}
          type={'undo'}
          simulationDetailForPlanMode={simulationDetailForPlanMode}
        />
        <_Menu
          className={clix(className?.redo)}
          type={'redo'}
          simulationDetailForPlanMode={simulationDetailForPlanMode}
        />
      </div>
    )
  },
)

const _getElementId = (index: number) => `PlanMenuSubMenuUndoRedo-${index}`

const _Menu = React.memo(
  ({
    className,
    type,
    simulationDetailForPlanMode,
  }: {
    className?: string
    type: 'undo' | 'redo'
    simulationDetailForPlanMode: SimulationInfoForPlanMode
  }) => {
    const { planParamsUndoRedoStack } = simulationDetailForPlanMode
    return (
      <ContextMenu2
        className={clix(className, ' disabled:lighten-2')}
        disabled={
          _processUndoRedoStack(planParamsUndoRedoStack)[type].length === 0
        }
        align="right"
      >
        <FontAwesomeIcon
          className=""
          icon={type === 'undo' ? faUndo : faRedo}
        />
        <_MenuItems
          type={type}
          simulationDetailForPlanMode={simulationDetailForPlanMode}
        />
      </ContextMenu2>
    )
  },
)

const _MenuItems = React.memo(
  ({
    type,
    simulationDetailForPlanMode,
  }: {
    type: 'undo' | 'redo'
    simulationDetailForPlanMode: SimulationInfoForPlanMode
  }) => {
    const { planParamsUndoRedoStack, setPlanParamsHeadIndex } =
      simulationDetailForPlanMode
    const [stack] = useState(
      () => _processUndoRedoStack(planParamsUndoRedoStack)[type],
    )
    const [hoverAtItem, setHoverAtItem] = useState(-1)
    const [popUpElement, setPopUpElement] = useState<HTMLDivElement | null>(
      null,
    )
    const [openTime] = useState(() => Date.now())
    const [moved, setMoved] = useState(false)
    const [hoverTopAndWidth, setHoverTopAndWidth] = useState<{
      top: number
      width: number
    } | null>(null)
    const [hoverBottom, setHoverBottom] = useState<{ bottom: number } | null>(
      null,
    )
    const updateHoverBottom = () => {
      if (hoverAtItem === -1) {
        setHoverBottom(null)
      } else {
        const itemi = fGet(
          window.document.getElementById(_getElementId(hoverAtItem)),
        )
        setHoverBottom({ bottom: itemi.offsetTop + itemi.offsetHeight })
      }
    }
    const updateHoverBottomRef = useRef(updateHoverBottom)
    updateHoverBottomRef.current = updateHoverBottom

    useEffect(() => updateHoverBottomRef.current(), [hoverAtItem])
    useEffect(() => {
      if (!popUpElement) return
      const observer = new ResizeObserver(() => {
        const item0 = window.document.getElementById(_getElementId(0))
        if (item0) {
          setHoverTopAndWidth({
            top: item0.offsetTop,
            width: item0.clientWidth - 16,
          })
        }
        updateHoverBottomRef.current()
      })
      observer.observe(popUpElement)
      return () => observer.disconnect()
    }, [popUpElement])
    return (
      <Menu.Items
        className="relative flex flex-col py-2 "
        ref={setPopUpElement}
      >
        <h2 className="text-xl font-bold px-5 py-2">
          {type === 'undo' ? 'Undo' : 'Redo'}
        </h2>
        <div
          className="absolute rounded-lg mx-2 border-2 border-gray-600 bg-gray-100 top-4 z-0 pointer-events-none duration-200"
          style={{
            transitionProperty: 'height, opacity',
            opacity: moved && hoverTopAndWidth && hoverBottom ? 1 : 0,
            top: `${hoverTopAndWidth?.top ?? 0}px`,
            width: `${hoverTopAndWidth?.width ?? 0}px`,
            height: `${
              moved && hoverTopAndWidth && hoverBottom
                ? hoverBottom.bottom - hoverTopAndWidth.top
                : 0
            }px`,
          }}
        />
        {stack.map((item, i) => (
          <_Item
            key={i}
            id={_getElementId(i)}
            className="relative z-10"
            item={item}
            onClick={() => {
              setPlanParamsHeadIndex(item.head)
            }}
            onMouseEnter={() => setHoverAtItem(i)}
            onMouseLeave={() => setHoverAtItem(-1)}
            onMouseMove={() => {
              if (Date.now() < openTime + 400) return
              setMoved(true)
            }}
          />
        ))}
      </Menu.Items>
    )
  },
)

type _UndoItem = {
  render: React.ReactNode
  timestamp: number
  head: number
  // baseURL: URL
  id: string
}

const _processUndoRedoStack = (
  undoRedoStack: WorkingPlanInfo['planParamsUndoRedoStack'],
  // planPaths: PlanPaths,
) => {
  const postProcess = (
    arr: {
      change: PlanParamsChangeAction
      head: number
      id: string
      params: PlanParams
      prevParams: PlanParams
    }[],
  ): _UndoItem[] => {
    return arr
      .map((x, i): _UndoItem => {
        const { render } = processPlanParamsChangeActionDeprecated(x.change)

        return {
          head: x.head,
          id: x.id,
          render: render(x.prevParams, x.params),
          timestamp: x.params.timestamp,
        }
      })
      .filter((x) => x.render !== null)
  }

  const undo = postProcess(
    _.takeRight(
      undoRedoStack.undos.map((x, i) => ({
        id: x.id,
        params: x.params,
        // i===0 will be skipped in the takeRight() via the Math.min().
        prevParams: i === 0 ? null : undoRedoStack.undos[i - 1].params,
        change: x.change,
        // We need i-1 because to undo the change described in action we have to
        // go to the previous state.  Note will be -1 for first item, but we
        // skip that item in the takeRight() via the Math.min()
        head: i - 1,
      })),
      // Since we removed first item don't have to filter "start"s which should only be in
      // the first element.
      Math.min(
        undoRedoStack.undos.length - 1,
        TARGET_UNDO_DEPTH - undoRedoStack.redos.length,
      ),
    )
      .map((x) => ({ ...x, prevParams: fGet(x.prevParams) }))
      .filter((x) => _filter(x.change)),
  ).reverse()

  const redo = postProcess(
    block(() => {
      const beforeFilter = undoRedoStack.redos.map((x, i) => ({
        id: x.id,
        params: x.params,
        prevParams:
          i === 0
            ? fGet(_.last(undoRedoStack.undos)).params
            : undoRedoStack.redos[i - 1].params,
        change: x.change,
        head: undoRedoStack.undos.length + i,
      }))
      const afterFilter = [] as typeof beforeFilter
      beforeFilter.forEach((x) => {
        if (_filter(x.change)) {
          afterFilter.push(x)
        } else {
          if (afterFilter.length !== 0) {
            afterFilter[afterFilter.length - 1].head = x.head
          }
        }
      })
      return afterFilter
    }),
  )

  return { undo, redo }
}

// Filter means we cannot get to the state right *before* the change.
const _filter = (change: PlanParamsChangeAction) =>
  change.type !== 'setDialogPosition' ||
  (change.value !== 'show-all-inputs' && change.value !== 'done')

const _Item = React.memo(
  ({
    id,
    item,
    onClick,
    onMouseEnter,
    onMouseLeave,
    onMouseMove,
    className,
  }: {
    id: string
    item: _UndoItem
    onClick: () => void
    onMouseEnter: () => void
    onMouseLeave: () => void
    onMouseMove: () => void
    className?: string
  }) => {
    const { currentTimestamp } = useSimulation()
    const { timestamp, render } = item

    const durationStr = useMemo(
      () => formatDistance(timestamp, currentTimestamp, { addSuffix: true }),
      [currentTimestamp, timestamp],
    )
    return (
      <Menu.Item
        as="button"
        id={id}
        className={'py-0.5'}
        onClick={onClick}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        onMouseMove={onMouseMove}
      >
        <div
          className={clix(className, 'mx-2 rounded-lg px-3 py-2  text-start')}
        >
          <h2>{render}</h2>
          <h2 className="text-xs lighten">{durationStr}</h2>
        </div>
      </Menu.Item>
    )
  },
)
