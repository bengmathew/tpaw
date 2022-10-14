import {faLink} from '@fortawesome/pro-light-svg-icons'
import {faCheck} from '@fortawesome/pro-regular-svg-icons'
import {faCopy} from '@fortawesome/pro-solid-svg-icons'
import {FontAwesomeIcon} from '@fortawesome/react-fontawesome'
import React, {useMemo, useState} from 'react'
import { tpawParamsForURL } from '../../App/UseTPAWParams'
import { useSimulation } from '../../App/WithSimulation'
import { ModalBase } from '../../Common/Modal/ModalBase'

export const PlanSummaryShare = React.memo(({className = ''}: {className?: string}) => {
  const [alertOpen, setAlertOpen] = useState(false)

  return (
    <>
      <button className={`${className}`} onClick={() => setAlertOpen(true)}>
        <FontAwesomeIcon className="mr-1 text-sm" icon={faLink} />
        Save
      </button>
      {alertOpen && <_Alert onClose={() => setAlertOpen(false)} />}
    </>
  )
})

const _Alert = React.memo(({onClose}: {onClose: () => void}) => {
  const {params} = useSimulation()
  const href = useMemo(() => {
    const url = new URL(window.location.href)
    url.searchParams.set('params', tpawParamsForURL(params))
    return url.toString()
  }, [params])
  const [isCopied, setIsCopied] = useState(false)

  const handleCopy = (onDone: () => void) =>()=> {
    void navigator.clipboard.writeText(href).then(() => {
      setIsCopied(true)
      window.setTimeout(onDone, 500)
      return null
    })
  }

  return (
    <ModalBase>
      {transitionOut => (
        <>
          <h2 className="text-lg font-bold mb-4">
            <FontAwesomeIcon icon={faLink} /> Save
          </h2>
          <div className="">
            <p className="">
              Use the following link to save your inputs as a bookmark or to
              share with anybody.
            </p>
            <input
              type="text"
              className="mt-4 cursor-pointer text-sm w-full border-b border-gray-500 border-dashed "
              readOnly
              value={href}
              onClick={handleCopy(()=>transitionOut(onClose))}
            />
          </div>
          <div className="flex justify-end mt-6 gap-x-4">
            <button className={`btn-md`} onClick={() => transitionOut(onClose)}>
              Cancel
            </button>
            <button
              className={`btn-md btn-dark flex gap-x-2 items-center`}
              disabled={isCopied}
              onClick={handleCopy(()=>transitionOut(onClose))}
            >
              {isCopied ? (
                <>
                  <FontAwesomeIcon icon={faCheck} />
                  <span className="">Copied!</span>
                </>
              ) : (
                <>
                  <FontAwesomeIcon icon={faCopy} />
                  <span className="">Copy</span>
                </>
              )}
            </button>
          </div>
        </>
      )}
    </ModalBase>
  )
})
