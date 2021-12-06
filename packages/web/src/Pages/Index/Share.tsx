import {faLink} from '@fortawesome/pro-light-svg-icons'
import {faCheck} from '@fortawesome/pro-regular-svg-icons'
import {faCopy, faTimes} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React, {useEffect, useMemo, useState} from 'react'
import {TPAWParams} from '../../TPAWSimulator/TPAWParams'
import {noCase} from '../../Utils/Utils'
import {tpawParamsForURL} from '../App/UseTPAWParams'

export const Share = React.memo(
  ({className = '', params}: {className?: string; params: TPAWParams}) => {
    const href = useMemo(() => {
      const url = new URL(window.location.href)
      url.searchParams.set('params', tpawParamsForURL(params))
      return url.toString()
    }, [params])
    const [state, setState] = useState<'idle' | 'copy' | 'copied'>('idle')
    useEffect(() => {
      if (state !== 'copied') return
      window.setTimeout(() => setState('idle'), 1000)
    }, [state])

    const handleCopy = () => {
      void navigator.clipboard.writeText(href).then(() => {
        setState('copied')
        return null
      })
    }
    return (
      <div className={`${className}`}>
        {state === 'idle' ? (
          <button
            className=""
            onClick={() => {
              setState('copy')
            }}
          >
            <FontAwesomeIcon className="mr-1 text-sm" icon={faLink} />
            Save as a link
          </button>
        ) : state === 'copy' ? (
          <div className="flex gap-x-4">
            <input
              type="text"
              className=" cursor-pointer text-sm w-[180px] border-b border-gray-500 border-dashed "
              readOnly
              value={href}
              onClick={handleCopy}
            />
            <button className="px-2" onClick={handleCopy}>
              <FontAwesomeIcon icon={faCopy} />
            </button>
            <button className="px-2" onClick={() => setState('idle')}>
            <FontAwesomeIcon icon={faTimes} />
            </button>
          </div>
        ) : state === 'copied' ? (
          <div className="">
            <FontAwesomeIcon icon={faCheck} /> Copied!
          </div>
        ) : (
          noCase(state)
        )}
      </div>
    )
  }
)
