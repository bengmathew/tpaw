import _ from 'lodash'
import {ReactNode, useLayoutEffect, useRef} from 'react'
import {useAssertConst} from '../../../../Utils/UseAssertConst'
import {fGet} from '../../../../Utils/Utils'

export const HeightsForWidths = <State extends string, Size>({
  sizing,
  children,
  onHeights,
}: {
  sizing: Record<State, Size>
  onHeights: (heights: Record<State, number>) => void
  children: (size: Size) => ReactNode
}) => {
  const states = _.keys(sizing) as State[]
  useAssertConst([onHeights, ...states])

  // eslint-disable-next-line react-hooks/rules-of-hooks
  const refs = states.map(state => useRef<HTMLDivElement>(null))

  useLayoutEffect(() => {
    const observer = new ResizeObserver(() => {
      onHeights(
        _.fromPairs(
          states.map((state, i) => [
            state,
            fGet(refs[i].current).getBoundingClientRect().height,
          ])
        ) as Record<State, number>
      )
    })
    states.map((s, i) => observer.observe(fGet(refs[i].current)))
    return () => observer.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <>
      {states.map((state, i) => (
        <div key={state} ref={refs[i]} className={`absolute invisible`}>
          {children(sizing[state])}
        </div>
      ))}
    </>
  )
}
