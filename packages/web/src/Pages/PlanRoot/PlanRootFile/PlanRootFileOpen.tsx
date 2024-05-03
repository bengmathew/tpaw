import React, { useRef } from 'react'
import clsx from 'clsx'
import { AppPage } from '../../App/AppPage'
import { mainPlanColors } from '../Plan/UsePlanColors'
import { fGet, getFullDatedDefaultPlanParams } from '@tpaw/common'
import { PlanFileData, PlanFileDataFns } from './PlanFileData'
import { useIANATimezoneName } from '../PlanRootHelpers/WithNonPlanParams'
import { errorToast } from '../../../Utils/CustomToasts'

export const PlanRootFileOpen = React.memo(
  ({
    onDone,
  }: {
    onDone: (filename: string | null, data: PlanFileData) => void
  }) => {
    const { ianaTimezoneName } = useIANATimezoneName()
    const inputRef = useRef<HTMLInputElement>(null)
    return (
      <AppPage title="Open File - TPAW Planner" className="h-screen" style={{}}>
        <div className="h-full flex flex-col justify-center items-center">
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            // eslint-disable-next-line @typescript-eslint/no-misused-promises
            onChange={async (e) => {
              const file = fGet(e.target.files)[0]
              const data = await PlanFileDataFns.open(file)
              if (!data) {
                errorToast('Not a valid TPAW file.')
                return
              }
              onDone(file.name, data)
            }}
          />
          <button
            className="  px-8 py-2  text-xl btn2-dark rounded-full "
            onClick={() => fGet(inputRef.current).click()}
          >
            Open File
          </button>
          <h2 className="my-2">or</h2>
          <button
            className={clsx('underline text-xl')}
            onClick={() =>
              onDone(
                null,
                PlanFileDataFns.getNew(
                  getFullDatedDefaultPlanParams(Date.now(), ianaTimezoneName),
                ),
              )
            }
          >
            Create a New Plan
          </button>
        </div>
      </AppPage>
    )
  },
)
