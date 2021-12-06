import {localPoint} from '@visx/event'
import _ from 'lodash'
import React, {useState} from 'react'
import {SVGSizing} from '../../../../Utils/SVG/UseSVGSizing'
import {fGet} from '../../../../Utils/Utils'

type Point = {x: number; y: number}

type _PointerState = {
  drag: {pointerID: number; value: number} | null
  focus: boolean
}

export type PointerXAxisDragResult = ReturnType<
  typeof usePointerXAxisDrag
>[number]
export function usePointerXAxisDrag({
  values,
  sizing,
  onDone,
}: {
  values: number[]
  sizing: SVGSizing
  onDone: (values: number[]) => void
}) {
  const {scale} = sizing

  const [pointerState, setPointerState] = useState<_PointerState[]>(
    values.map(() => ({drag: null, focus: false}))
  )

  const transform = (values: number[], i: number) => {
    if (values.length === 1) return [Math.round(values[0])]
    // FEATURE: the +-1 should be generalized through parameters.
    const clamped =
      i === 0
        ? Math.min(values[i], values[i + 1] - 1)
        : i === values.length - 1
        ? Math.max(values[i - 1] + 1, values[i])
        : _.clamp(values[i], values[i - 1] + 1, values[i + 1] - 1)

    return _withIChanged(values, i, Math.round(clamped))
  }

  const pointToValue = (point: Point) =>
    _.clamp(scale.x.invert(point.x), 0, sizing.xDomain.end)

  const pointerIndex = pointerState.findIndex(x => x.drag !== null)
  const displayValues = transform(
    pointerState.map((x, i) => x.drag?.value ?? values[i]),
    pointerIndex === -1 ? 0 : pointerIndex
  )

  return pointerState.map((currPointerState, i) => ({
    props: _getProps(
      (d: 1 | -1) => _.clamp(values[i] + d, 0, sizing.xDomain.end),
      pointToValue,
      currPointerState,
      x => setPointerState(_withIChanged(pointerState, i, x)),
      value => onDone(transform(_withIChanged(displayValues, i, value), i))
    ),
    displayValue: displayValues[i],
    inDrag: currPointerState.drag !== null,
    focus: currPointerState.focus,
  }))
}

function _getProps(
  delta: (x: 1 | -1) => number,
  pointToValue: (point: Point) => number,
  pointerState: _PointerState,
  setPointerState: (x: _PointerState) => void,
  onDone: (value: number) => void
) {
  const getValue = (e: React.PointerEvent) => pointToValue(fGet(localPoint(e)))
  const dragState = pointerState.drag
  const setDragState = (drag: _PointerState['drag']) =>
    setPointerState({...pointerState, drag})
  const setFocus = (focus: boolean) => setPointerState({...pointerState, focus})

  const stop = (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!dragState) return
    if (dragState.pointerID !== e.pointerId) return false
    ;(e.target as Element).releasePointerCapture(e.pointerId)

    setDragState(null)
    return true
  }

  return {
    onPointerDown(e: React.PointerEvent) {
      if (dragState) return
      ;(e.target as Element).setPointerCapture(e.pointerId)
      setDragState({pointerID: e.pointerId, value: getValue(e)})
    },

    onPointerMove(e: React.PointerEvent) {
      if (!dragState) return
      if (dragState.pointerID !== e.pointerId) return
      setDragState({pointerID: e.pointerId, value: getValue(e)})
    },

    onPointerUp(e: React.PointerEvent) {
      if (stop(e)) onDone(getValue(e))
    },
    onPointerCancel: stop,
    onPointerLeave: stop,

    onKeyDown(e: React.KeyboardEvent) {
      switch (e.key) {
        case 'ArrowRight':
        case 'ArrowUp':
          onDone(delta(+1))
          break
        case 'ArrowLeft':
        case 'ArrowDown':
          onDone(delta(-1))
          break
      }
    },

    tabIndex: 0,
    onFocus: () => setFocus(true),
    onBlur: () => setFocus(false),
  }
}

function _withIChanged<T>(arr: T[], i: number, v: T) {
  const result = [...arr]
  result[i] = v
  return result
}
