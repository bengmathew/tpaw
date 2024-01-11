import { useRouter } from 'next/router'
import React, { useEffect, useRef } from 'react'
import { appPaths } from '../../../AppPaths'
import { useFirebaseUser } from '../../App/WithFirebaseUser'
import { WithUser } from '../../App/WithUser'
import { SimulationParams } from '../PlanRootHelpers/WithSimulation'
import { PlanRootLocalImpl } from './PlanRootLocalImpl'

export const PlanRootLocalMain = React.memo(
  ({ pdfReportInfo }: { pdfReportInfo: SimulationParams['pdfReportInfo'] }) => {
    const isLoggedIn = useFirebaseUser() !== null

    const router = useRouter()
    const handleIsLoggedIn = () => {
      const url = appPaths.plan()
      void router.push(url, url, { shallow: true })
    }
    const handleIsLoggedInRef = useRef(handleIsLoggedIn)
    handleIsLoggedInRef.current = handleIsLoggedIn
    useEffect(() => {
      if (!isLoggedIn) return
      handleIsLoggedInRef.current()
    }, [isLoggedIn])

    const [key, setKey] = React.useState(0)

    if (isLoggedIn) return <></>
    return (
      <WithUser userFragmentOnQueryKey={null} key={key}>
        <PlanRootLocalImpl
          reload={() => setKey((x) => x + 1)}
          pdfReportInfo={pdfReportInfo}
        />
      </WithUser>
    )
  },
)
