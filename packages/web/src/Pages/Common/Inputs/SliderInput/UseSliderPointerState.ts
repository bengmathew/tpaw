import { localPoint } from '@visx/event'
import React, { useState } from 'react'
import { fGet } from '../../../../Utils/Utils'
import { SliderSizing } from './UseSliderSizing'

export type SliderPointerState = {
  drag: { pointerId: number; index: number } | null
  focus: boolean
}

export function useSliderPointerState(
  index: number,
  sizing: SliderSizing,
  onChange: (index: number) => void,
) {
  const { pixelXToDataX, numPoints } = sizing

  const [drag, setDrag] = useState<SliderPointerState['drag']>(null)
  const [focus, setFocus] = useState(false)

  const stop = (e: React.PointerEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!drag) return
    if (drag.pointerId !== e.pointerId) return false
    ;(e.target as Element).releasePointerCapture(e.pointerId)

    setDrag(null)
    return true
  }

  const pointerEventToIndex = (e: React.PointerEvent) =>
    Math.round(pixelXToDataX(fGet(localPoint(e)).x))

  return {
    pointerState: { drag, focus },
    propsForPointerHTMLElement: {
      onPointerDown(e: React.PointerEvent) {
        if (drag) return
        ;(e.target as Element).setPointerCapture(e.pointerId)
        setDrag({ pointerId: e.pointerId, index: pointerEventToIndex(e) })
      },

      onPointerMove(e: React.PointerEvent) {
        if (!drag) return
        if (drag.pointerId !== e.pointerId) return
        setDrag({ pointerId: e.pointerId, index: pointerEventToIndex(e) })
      },

      onPointerUp(e: React.PointerEvent) {
        if (stop(e)) onChange(pointerEventToIndex(e))
      },
      onPointerCancel: stop,
      onPointerLeave: stop,

      onKeyDown(e: React.KeyboardEvent) {
        switch (e.key) {
          case 'ArrowRight':
            onChange(Math.min(numPoints - 1, index + 1))
            break
          case 'ArrowLeft':
            onChange(Math.max(0, index - 1))
            break
        }
      },

      tabIndex: 0,
      onFocus: () => setFocus(true),
      onBlur: () => setFocus(false),
    },
  }
}
