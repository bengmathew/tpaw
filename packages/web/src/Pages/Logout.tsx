import { getAuth, signOut } from 'firebase/auth'
import React, { useEffect } from 'react'
import { asyncEffect } from '../Utils/AsyncEffect'
import { useURLParam } from '../Utils/UseURLParam'
import { useURLUpdater } from '../Utils/UseURLUpdater'
import { AppPage } from './App/AppPage'
import { appPaths } from '../AppPaths'

export const Logout = React.memo(() => {
  const urlUpdater = useURLUpdater()
  const dest = useURLParam('dest') ?? appPaths.plan()
  useEffect(() => {
    return asyncEffect(async () => {
      await signOut(getAuth())
    })
  }, [dest, urlUpdater])
  return (
    <AppPage
      title="Logout - TPAW Planner"
      className="h-screen flex flex-col justify-center items-center"
    >
      <div className="">
        <h2 className="text-lg">You have been logged out.</h2>
      </div>
    </AppPage>
  )
})
