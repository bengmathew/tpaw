import React, { useEffect } from 'react'
import { appPaths } from '../../src/AppPaths'
import { useFirebaseUser } from '../../src/Pages/App/WithFirebaseUser'
import { PlanRoot } from '../../src/Pages/PlanRoot/PlanRoot'
import { planRootGetStaticPaths } from '../../src/Pages/PlanRoot/PlanRootHelpers/PlanRootGetStaticPaths'
import {
  planRootGetStaticProps,
  PlanStaticProps,
} from '../../src/Pages/PlanRoot/PlanRootHelpers/PlanRootGetStaticProps'
import { PlanRootLoginOrLocal } from '../../src/Pages/PlanRoot/PlanRootLoginOrLocal'
import { useURLParam } from '../../src/Utils/UseURLParam'
import { useURLUpdater } from '../../src/Utils/UseURLUpdater'

export default React.memo(({ planContent, marketData }: PlanStaticProps) => {
  const firebaseUser = useFirebaseUser()

  const urlUpdater = useURLUpdater()
  const redirectToLink = !!useURLParam('params')
  useEffect(() => {
    if (!redirectToLink) return
    const url = appPaths.link()
    new URL(window.location.href).searchParams.forEach((value, key) =>
      url.searchParams.set(key, value),
    )
    urlUpdater.replace(url)
  }, [redirectToLink, urlUpdater])
  if (redirectToLink) return <></>

  if (!firebaseUser) return <PlanRootLoginOrLocal />
  return (
    <PlanRoot
      planContent={planContent}
      marketData={marketData}
      src={{ type: 'serverMain' }}
    />
  )
})

export const getStaticProps = planRootGetStaticProps
export const getStaticPaths = planRootGetStaticPaths
