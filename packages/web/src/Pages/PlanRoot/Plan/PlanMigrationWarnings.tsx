import React, { useState } from 'react'
import { CenteredModal } from '../../Common/Modal/CenteredModal'
import { useSimulation } from '../PlanRootHelpers/WithSimulation'

export const PlanMigrationWarnings = React.memo(() => {
  const { planMigratedFromVersion, updatePlanParams } = useSimulation()

  const [migrations, setMigrations] = useState(() => {
    const result = [] as { title: string; message: string }[]
    if (planMigratedFromVersion < 17) {
      result.push({
        title: 'Migration to New Risk Inputs',
        message: `The planner has been updated to use different inputs for risk.
          Your previous risk inputs have been migrated to the new version.
          The mapping is not exact, so please review the inputs in the risk
          section.`,
      })
    }
    if (planMigratedFromVersion < 20) {
      result.push({
        title: 'Migration to Calendar Inputs',
        message: `The planner now uses calendar dates for time based inputs. Your
        inputs have been migrated as needed. As part of the migration,
        some dates were approximated. Please review all the time based
        entries (age, pension start dates, etc.) to make sure that it is
        correct.`,
      })
    }
    return result
  })

  return (
    <CenteredModal
      className=" dialog-outer-div"
      show={migrations.length > 0}
      onOutsideClickOrEscape={null}
    >
      {migrations.length === 1 ? (
        <>
          <h2 className=" dialog-heading">{migrations[0].title}</h2>
          <div className=" dialog-content-div">
            <p className="p-base">{migrations[0].message}</p>
          </div>
        </>
      ) : (
        <>
          <h2 className=" dialog-heading">Migration to New Inputs</h2>
          <div className=" dialog-content-div">
            <p className="p-base">
              The planner has been updated to use different inputs. Your plan
              has been migrated as follows
            </p>
          </div>
          {migrations.map(({ title, message }, i) => (
            <div key={i} className="mt-4">
              <h2 className="font-semibold">{title}</h2>
              <p className="p-base mt-2">{message}</p>
            </div>
          ))}
        </>
      )}
      <div className=" dialog-button-div">
        <button
          className=" dialog-button-dark"
          onClick={() => {
            setMigrations([])
            updatePlanParams('noOpToMarkMigration', null)
          }}
        >
          Close
        </button>
      </div>
    </CenteredModal>
  )
})
