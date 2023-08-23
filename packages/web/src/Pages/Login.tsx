import React, { useEffect } from 'react'
import { useURLParam } from '../Utils/UseURLParam'
import { useURLUpdater } from '../Utils/UseURLUpdater'
import { Config } from './Config'
import { appPaths } from '../AppPaths'

export const Login = React.memo(() => {
  const urlUpdater = useURLUpdater()
  const dest = useURLParam('dest') ?? appPaths.plan()
  useEffect(() => {
    urlUpdater.replace(dest)
  }, [dest, urlUpdater])
  return <></>
})
