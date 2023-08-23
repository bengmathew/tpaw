import { block } from './Utils'

export const getAppPaths = (base: URL) => {
  const urlFromPath = (path: string) => new URL(path, base)

  const planFromPath = (path: string) =>
    block(() => {
      const result = () => urlFromPath(path)
      result['help'] = () => urlFromPath(`${path}/help`)
      result['age'] = () => urlFromPath(`${path}/age`)
      result['current-portfolio-balance'] = () =>
        urlFromPath(`${path}/current-portfolio-balance`)
      result['future-savings'] = () => urlFromPath(`${path}/future-savings`)
      result['income-during-retirement'] = () =>
        urlFromPath(`${path}/income-during-retirement`)
      result['extra-spending'] = () => urlFromPath(`${path}/extra-spending`)
      result['legacy'] = () => urlFromPath(`${path}/legacy`)
      result['risk'] = () => urlFromPath(`${path}/risk`)
      result['spending-ceiling-and-floor'] = () =>
        urlFromPath(`${path}/spending-ceiling-and-floor`)
      result['strategy'] = () => urlFromPath(`${path}/strategy`)
      result['expected-returns'] = () => urlFromPath(`${path}/expected-returns`)
      result['inflation'] = () => urlFromPath(`${path}/inflation`)
      result['simulation'] = () => urlFromPath(`${path}/simulation`)
      result['dev-misc'] = () => urlFromPath(`${path}/dev-misc`)
      result['dev-simulations'] = () => urlFromPath(`${path}/dev-simulations`)
      result['dev-historical-returns'] = () =>
        urlFromPath(`${path}/dev-historical-returns`)
      result['dev-additional-spending-tilt'] = () =>
        urlFromPath(`${path}/dev-additional-spending-tilt`)
      result['dev-time'] = () => urlFromPath(`${path}/dev-time`)
      return result
    })

  return {
    account: () => urlFromPath('/account'),
    learn: () => urlFromPath('/learn'),
    logout: () => urlFromPath('/logout'),
    login: (destURL: URL) =>
      urlFromPath(
        `/login?${new URLSearchParams({
          dest: `${destURL.pathname}${destURL.search}`,
        }).toString()}`,
      ),
    guest: planFromPath('/guest'),
    link: planFromPath('/link'),
    plans: () => urlFromPath('/plans'),
    plan: planFromPath('/plan'),
    'alt-plan': planFromPath('/alt-plan'),
  }
}

export type PlanPaths = ReturnType<typeof getAppPaths>['plan']
