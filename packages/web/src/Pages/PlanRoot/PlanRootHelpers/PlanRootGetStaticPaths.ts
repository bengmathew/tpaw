import { paramsInputTypes } from '../Plan/PlanInput/Helpers/PlanInputType'

export const planRootGetStaticPaths = () => ({
  paths: [
    ...paramsInputTypes.map((section) => ({ params: { section: [section] } })),
    { params: { section: ['help'] } },
    { params: { section: ['print'] } },
    { params: { section: null } },
  ],
  fallback: false,
})
