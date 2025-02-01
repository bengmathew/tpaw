import { faCheck } from '@fortawesome/pro-regular-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import clsx from 'clsx'
import { useEffect, useState } from 'react'
import { assert } from '../../../Utils/Utils'
import { CenteredModal } from '../../Common/Modal/CenteredModal'
import { useSimulationResultInfo } from '../PlanRootHelpers/WithSimulation'
import { usePlanColors } from './UsePlanColors'



const TIMEOUT_MS = 10000

export const PlanSimulationDelayModal = () => {
  const { simulationIsRunningInfo } = useSimulationResultInfo()

  // Storing the info to display instead of a boolean flag to track when to show
  // and using the info directly from the context is to freeze the info for the
  // _Content component through the show/hide transition.
  const [frozenResultInfo, setFrozenResultInfo] = useState<{
    simulationStartTimestamp: {
      countingFromThisSimulation: number
      countingFromTheFirstDebouncedSimulation: number
    }
  } | null>(null)
  const [show, setShow] = useState(false)

  useEffect(() => {
    if (simulationIsRunningInfo.isRunning) {
      const timeout = setTimeout(
        () => {
          setFrozenResultInfo({
            simulationStartTimestamp:
              simulationIsRunningInfo.simulationStartTimestamp,
          })
          setShow(true)
        },
        Math.max(
          1,
          TIMEOUT_MS -
            (Date.now() -
              simulationIsRunningInfo.simulationStartTimestamp
                .countingFromTheFirstDebouncedSimulation),
        ),
      )
      return () => clearTimeout(timeout)
    }
  }, [simulationIsRunningInfo])

  return (
    <CenteredModal
      className=" dialog-outer-div"
      show={show}
      onOutsideClickOrEscape={null}
    >
      <Content
        simulationIsRunningInfo={frozenResultInfo}
        onClose={
          simulationIsRunningInfo.isRunning ? null : () => setShow(false)
        }
      />
    </CenteredModal>
  )
}

const Content = ({
  onClose,
  simulationIsRunningInfo,
}: {
  simulationIsRunningInfo: {
    simulationStartTimestamp: {
      countingFromThisSimulation: number
      countingFromTheFirstDebouncedSimulation: number
    }
  } | null
  onClose: (() => void) | null
}) => {
  assert(simulationIsRunningInfo)

  const [currentTimestamp, setCurrentTimestamp] = useState(Date.now())

  useEffect(() => {
    const abortController = new AbortController()
    const callback = () => {
      if (!abortController.signal.aborted) {
        requestAnimationFrame(callback)
        setCurrentTimestamp(Date.now())
      }
    }
    requestAnimationFrame(callback)
    return () => abortController.abort()
  }, [])

  const seconds = Math.floor(
    (currentTimestamp -
      simulationIsRunningInfo.simulationStartTimestamp
        .countingFromTheFirstDebouncedSimulation) /
      1000,
  )
  const { shades } = usePlanColors()

  return (
    <>
      <h1 className="dialog-heading">Waiting for Results...</h1>
      <div className=" dialog-content-div">
        <p className=" p-base mt-4">
          The results are taking longer than expected to compute. Please wait
          for the results before continuing.
        </p>

        <p className=" p-base mt-2">
          Status:{' '}
          {onClose ? (
            <span className=" text-successFG">
              Completed <FontAwesomeIcon icon={faCheck} />
            </span>
          ) : (
            <span>
              Running for{' '}
              <span className="font-mono text-[16px]">{seconds}</span> seconds
            </span>
          )}
        </p>
      </div>
      <div className="dialog-button-div ">
        <button
          className={clsx(
            'dialog-button-dark w-[120px]',
            onClose === null && 'lighten-2',
          )}
          disabled={onClose === null}
          onClick={onClose ?? undefined}
        >
          {onClose ? 'Close' : 'Waiting...'}
        </button>
      </div>
    </>
  )
}
