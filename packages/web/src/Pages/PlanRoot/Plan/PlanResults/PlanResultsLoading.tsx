import { block, noCase } from '@tpaw/common'
import { useEffect, useState } from 'react'
import { usePlanColors } from '../UsePlanColors'

const DELAY_MS = 500
type _Stage = 'loading' | 'loaded' | 'hiding' | 'hidden'
export const PlanResultsLoading = ({
  showText,
  isRunning,
  dontHide,
}: {
  showText: boolean
  isRunning: boolean
  dontHide: boolean
}) => {
  const [state, setState] = useState<{
    stage: _Stage
    stageStartedTimestamp: number
  }>({ stage: 'hidden', stageStartedTimestamp: Date.now() })

  useEffect(() => {
    const abortController = new AbortController()
    const signal = abortController.signal
    const now = Date.now()

    const getTransitionStep =
      (
        targetStage: _Stage,
        previousStageDuration: number,
        next: null | ((previousStageStartTimestamp: number) => void),
      ) =>
      (previousStageStartTimestamp: number) =>
        window.setTimeout(
          () => {
            if (signal.aborted) return
            const now = Date.now()
            setState({ stage: targetStage, stageStartedTimestamp: now })
            next?.(now)
          },
          Math.max(
            0,
            previousStageDuration - (now - previousStageStartTimestamp),
          ),
        )
    const toHiddenFrom = block(() => {
      const hidden = () => {}
      const hiding = getTransitionStep('hidden', 500, hidden)
      const loaded = getTransitionStep('hiding', 500, hiding)
      const loading = getTransitionStep('loaded', 500, loaded)
      return {
        hidden,
        hiding,
        loaded,
        loading,
      }
    })

    if (isRunning) {
      // This might in principle not be accurate if useEffect gets called
      // multiple times, but should be rare enough and it is hard to account for
      // that.
      toHiddenFrom[state.stage](state.stageStartedTimestamp)
      window.setTimeout(() => {
        if (signal.aborted) return
        setState({ stage: 'loading', stageStartedTimestamp: Date.now() })
      }, DELAY_MS)
    } else {
      toHiddenFrom[state.stage](state.stageStartedTimestamp)
    }
    return () => abortController.abort()

    // disabling stage dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRunning])

  // "hiding" and 'hidden' are handled here, not in the children.
  const stageForChildren =
    state.stage === 'hiding'
      ? 'loaded'
      : state.stage === 'hidden'
        ? 'loading'
        : state.stage
  if (!dontHide && state.stage === 'hidden') return <></>
  return (
    <div
      style={{
        transitionProperty: 'opacity',
        transitionDuration: '500ms',
        opacity: state.stage === 'hiding' && !dontHide ? 0 : 1,
      }}
    >
      <PingPong stage={stageForChildren} />
      {showText && <_Text stage={stageForChildren} />}
    </div>
  )
}

const _Text = ({ stage }: { stage: 'loading' | 'loaded' }) => {
  return (
    <h2 className="text-[11px] -mt-[1px] lighten text-right">
      {stage === 'loading'
        ? 'calculating...'
        : stage === 'loaded'
          ? 'completed'
          : noCase(stage)}
    </h2>
  )
}

const PingPong = ({ stage }: { stage: 'loading' | 'loaded' }) => {
  const planColors = usePlanColors()
  return (
    <div
      className="relative w-full h-[5px] rounded-full overflow-hidden "
      style={{ backgroundColor: planColors.shades.alt[10].hex }}
    >
      <div
        className="absolute  h-full rounded-full"
        style={{
          transitionProperty: 'width',
          transitionDuration: '500ms',
          width:
            stage === 'loaded'
              ? '100px'
              : stage === 'loading'
                ? '20%'
                : noCase(stage),

          backgroundColor: planColors.shades.main[8].hex,
          animationDirection: 'alternate-reverse',
          animationIterationCount: 'infinite',
          animationTimingFunction: 'ease-in-out',
          animationName: 'slide-left-right',
          animationDuration: '500ms',
        }}
      />
    </div>
  )
}
