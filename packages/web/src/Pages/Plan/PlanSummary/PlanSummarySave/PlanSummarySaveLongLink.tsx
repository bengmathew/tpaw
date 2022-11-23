import {
  faCheck,
  faClipboard
} from '@fortawesome/pro-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import React, { useState } from 'react'
import { useSimulation } from '../../../App/WithSimulation'
import { Config } from '../../../Config'




export const PlanSummarySaveLongLink = React.memo(({ className = '' }: { className?: string} ) => {
  const [copied, setCopied] = useState(false)
  const { params } = useSimulation()
  return (
    <button
      className={`${className}`}
      onClick={() => {
        const href = new URL(
          Config.client.urls.app(
            `/plan?${new URLSearchParams({
              params: JSON.stringify(params),
            }).toString()}`
          )
        ).toString()

        void navigator.clipboard.writeText(href).then(() => {
          setCopied(true)
          window.setTimeout(() => setCopied(false), 1000)
          return null
        })
      }}
    >
      {copied ? (
        <>
          <FontAwesomeIcon className="mr-2" icon={faClipboard} />Copied to
          clipboard{' '}
          <FontAwesomeIcon className="ml-1 font-bold" icon={faCheck} />
        </>
      ) : (
        'Long Link'
      )}
    </button>
  )
})
