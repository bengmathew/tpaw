import { faHistory } from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { Menu } from '@headlessui/react'
import { assertFalse, noCase } from '@tpaw/common'
import clix from 'clsx'
import React, { useEffect, useRef, useState } from 'react'
import { useURLUpdater } from '../../../../../Utils/UseURLUpdater'
import { useSimulation } from '../../../PlanRootHelpers/WithSimulation'

export const PlanMenuActionViewPlanHistory = React.memo(
  ({
    className,
    historyStatus,
  }: {
    className?: string
    historyStatus: 'fetching' | 'fetched' | 'failed'
  }) => {
    const [waitForHistory, setWaitForHistory] = useState(false)
    const { currentTimestamp } = useSimulation()

    const handleSwitchToHistoryMode = () => {
      const url = new URL(window.location.href)
      url.searchParams.set('rewindTo', `${currentTimestamp}`)
      urlUpdater.push(url)
    }
    const handleSwitchToHistoryModeRef = useRef(handleSwitchToHistoryMode)
    handleSwitchToHistoryModeRef.current = handleSwitchToHistoryMode

    const urlUpdater = useURLUpdater()
    useEffect(() => {
      if (!waitForHistory) return
      switch (historyStatus) {
        case 'fetched':
          handleSwitchToHistoryModeRef.current()
        case 'failed':
          assertFalse()
        case 'fetching':
          break
        default:
          noCase(historyStatus)
      }
    }, [historyStatus, urlUpdater, waitForHistory])

    return (
      <Menu.Item
        as="button"
        className={clix(className)}
        onClick={(e) => {
          if (historyStatus === 'fetched') {
            handleSwitchToHistoryMode()
          } else {
            // This keeps the menu open (only  on click through, not on keyboard)
            // As of Jun 2023, no solution for keyboard:
            // https://github.com/tailwindlabs/headlessui/discussions/1122
            e.preventDefault()
            setWaitForHistory(true)
          }
        }}
      >
        <span className="inline-block w-[25px]">
          <FontAwesomeIcon icon={faHistory} />
        </span>{' '}
        {waitForHistory ? 'Fetching History...' : 'View Plan History'}
      </Menu.Item>
    )
  },
)
