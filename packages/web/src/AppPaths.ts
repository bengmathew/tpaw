import { getAppPaths } from '@tpaw/common'
import { Config } from './Pages/Config'

export const appPaths = getAppPaths(new URL(Config.client.urls.app()))
