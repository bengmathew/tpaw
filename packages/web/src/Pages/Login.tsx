import React, { useEffect } from 'react'
import { useURLParam } from '../Utils/UseURLParam'
import { useURLUpdater } from '../Utils/UseURLUpdater'
import { Config } from './Config'

export const Login = React.memo(() => {
  const urlUpdater = useURLUpdater()
  const dest = useURLParam('dest') ?? '/plan'
  useEffect(() => {
    urlUpdater.replace(dest)
  }, [dest, urlUpdater])
  return <></>
})

export const loginPath = () => {
  let currentURL = new URL(window.location.href)
  if (!currentURL.pathname.startsWith('/plan')) {
    currentURL = new URL(Config.client.urls.app('/plan'))
  }
  const dest = `${currentURL.pathname}${currentURL.search}`
  return `/login?${new URLSearchParams({ dest }).toString()}`
}
