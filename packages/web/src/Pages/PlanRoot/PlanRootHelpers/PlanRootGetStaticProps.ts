import { MarketData } from '@tpaw/common'
import { GetStaticProps } from 'next'
import { Contentful } from '../../../Utils/Contentful'
import { getMarketData } from '../../Common/GetMarketData'

type _FetchedInline = Awaited<ReturnType<typeof Contentful.fetchInline>>
type _Guide = { guide: _FetchedInline }
type _Intro = { intro: _FetchedInline }
type _Menu = { menu: _FetchedInline }
type _IntroAndGuide = _Guide & _Intro
type _IntroAndGuideAndMenu = _IntroAndGuide & _Menu

export type PlanContent = {
  age: {
    introRetired: _FetchedInline
    introNotRetired: _FetchedInline
  }
  'current-portfolio-balance': _Intro
  'future-savings': _Intro
  'income-during-retirement': _Intro
  'extra-spending': _Intro & {
    essential: _FetchedInline
    discretionary: _FetchedInline
  }
  legacy: {
    introAmount: _FetchedInline
    introAssets: _FetchedInline
  }
  'spending-ceiling-and-floor': {
    ceiling: _FetchedInline
    floor: _FetchedInline
  }
  risk: _Intro
  // 'stock-allocation': _IntroAndGuide
  // 'spending-tilt': _IntroAndGuide
  // lmp: _IntroAndGuide
  withdrawal: _IntroAndGuide
  strategy: _IntroAndGuide & {
    cardIntro: {
      TPAW: _FetchedInline
      SPAW: _FetchedInline
      SWR: _FetchedInline
    }
    rewardRiskRatioIntro: _FetchedInline
  }
  'expected-returns-and-volatility': _IntroAndGuide
  inflation: _IntroAndGuide
  simulation: {
    guide: _FetchedInline
  }
  history: Record<string, never>
  'dev-misc': Record<string, never>
  'dev-simulations': Record<string, never>
  'dev-time': Record<string, never>
  misc: {
    realDollarExplanation: _FetchedInline
  }
  chart: {
    realBlurb: _FetchedInline
    spending: {
      total: _IntroAndGuideAndMenu
      regular: _IntroAndGuideAndMenu
      discretionary: _IntroAndGuideAndMenu
      essential: _IntroAndGuideAndMenu
    }
    portfolio: _IntroAndGuideAndMenu
    'asset-allocation-savings-portfolio': _IntroAndGuideAndMenu
    'asset-allocation-total-portfolio': _IntroAndGuideAndMenu
    withdrawalRate: _IntroAndGuideAndMenu
    rewardRiskRatio: _IntroAndGuide
  }
  help: _FetchedInline
}

export type PlanStaticProps = {
  marketData: MarketData.Data
  planContent: PlanContent
}

export const planRootGetStaticProps: GetStaticProps<
  PlanStaticProps
> = async () => {
  const [marketData, planContent] = await Promise.all([
    getMarketData(),
    _getPlanContent(),
  ])
  return {
    props: {
      marketData,
      planContent,
    },
  }
}

const _getPlanContent = async () =>
  await Contentful.fetchInlineMultiple({
    age: {
      introRetired: '1dZPTbtQfLz3cyDrMGrAQB',
      introNotRetired: '43EyTxBVHWOPcA6rgBpsnG',
      // guide: ('5EtkcdtSIg0rS8AETEsgnm'),
    },
    'current-portfolio-balance': {
      intro: '3iLyyrQAhHnzuc4IdftWT3',
      // guide: ('5RE7wTwvtTAsF1sWpKrFW2'),
    },
    'future-savings': {
      intro: '2rPr5mMTcScftXhletDeb4',
      // guide: ('5aJN2Z4tZ7zQ6Tw69VelRt'),
    },
    'income-during-retirement': {
      intro: '3OqUTPDVRGzgQcVkJV7Lew',
      // guide: ('1MHvhL8ImdOL9FxE5qxK6F'),
    },
    'extra-spending': {
      intro: '01kv7sKzniBagrcIwX86tJ',
      essential: '2SFPqFDpLMmxO1VTOK8Fuh',
      discretionary: '5pwXrjG0Vu1SkMrDqqrrsC',
      // guide: ('5zDvtk4dDOonIkoIyOeQH8'),
    },
    legacy: {
      introAmount: 'aSdQuriQu9ztfs812MRJj',
      introAssets: '5glA8ryQcNh7SHP9ZlkZ2y',
      // guide: ('5nCHpNy6ReAEtBQTvDTwBf'),
    },
    'spending-ceiling-and-floor': {
      ceiling: '5qYghkV44aNpw9ynVl6UBN',
      floor: '19Llaw2GVZhEfBTfGzE7Ns',
    },

    risk: {
      intro: '5ROsQ76o7dv8rpmMCiuCMd',
    },
    // 'stock-allocation': {
    //   guide: '3ofgPmJFLgtJpjl26E7jpB',
    //   intro: 'xWXcgVScUfdK1PaTNQeKz',
    // },
    // 'spending-tilt': {
    //   guide: '6Dv02w4fUuFQUjyWxnR7Vq',
    //   intro: '4UwuCPjuTz3SbwUcZIrLEG',
    // },
    // lmp: {
    //   guide: '3ofgPmJFLgtJpjl26E7jpB',
    //   intro: '5FiPQS04F4uFngEMJium3B',
    // },
    withdrawal: {
      guide: '7eGRhX0KpxK2wKCzDTHLOs',
      intro: '3H8rgiVzmnyD6H3ZUjvgp8',
    },
    strategy: {
      intro: '52f9yaDqUCBBg3mkqGdZPc',
      cardIntro: {
        TPAW: '4qYue9K3cSpEkSrAhIn7AV',
        SPAW: '5W26KpQeXY9nC3FgKioesF',
        SWR: '3dQbwDNqEarfXVOUnZjJn9',
      },
      rewardRiskRatioIntro: '7wNIfORQHqumvZG6wWcmqG',
      guide: '5F0tZKpZ2SPvljHIGkPYmy',
    },
    'expected-returns-and-volatility': {
      intro: '2NxIclWQoxuk0TMVH0GjhR',
      guide: '2GxHf6q4kfRrz6AnFLniFh',
    },
    inflation: {
      intro: '76BgIpwX9yZetMGungnfwC',
      guide: '6LqbR3PBA1uDe9xU2V1hk9',
    },
    simulation: {
      guide: '5alyO5geIHnQsw8ZMpbyf5',
    },
    history: {},
    'dev-misc': {},
    'dev-simulations': {},
    'dev-time': {},
    misc: {
      realDollarExplanation: '3Xp6QN75C8mEljylz013Ek',
    },
    chart: {
      realBlurb: '0OBu1kF01M6HktVFh6haL',
      spending: {
        total: {
          intro: '6MH8oPq7ivMYJ4Ii8ZMtwg',
          guide: '4tlCDgSlcKXfO8hfmZvBoF',
          menu: '21U0B92yz78PBR2YlX1CJP',
        },
        regular: {
          intro: '3e3y2gADSUszUxSqWvC2EV',
          guide: '14KuXhGuaRok2dW0ppyArU',
          menu: '2tmRByrkc7dPNVOS63CUy3',
        },
        discretionary: {
          intro: '3pcwRLnI1BaBm0Hqv0Zyvg',
          guide: '4QRDrEzc3uJHNI4SzRJ6r7',
          menu: '7F7JGrPs7ShQ2G4GyEkJ75',
        },
        essential: {
          intro: '3oWMTqbZARYjiZ2G9fuwTb',
          guide: '4Vd9Ay5NnEodIMiBl83Vfs',
          menu: '7xqfxGtRc4pHZFJvR9zaY1',
        },
      },
      portfolio: {
        intro: '5KzxtC01WrdXnahFd98zet',
        guide: '2iQeojVfV3Fw18WV4TRATR',
        menu: '7cRJH6TdeEPpcom84B8Dch',
      },
      'asset-allocation-savings-portfolio': {
        intro: '247ji3WlKEyiMcthzbcaUX',
        guide: '1r6sBwyo6ulnzLQwQOM05L',
        menu: '4OsYFfjGHmGn8HucKv2thb',
      },
      'asset-allocation-total-portfolio': {
        intro: '1uowfKS8e8RqIANAucKTkX',
        guide: '1k57vBTerLEOp5ADAxDxoV',
        menu: '8MHwQEcXljZlEIcZrQRKd',
      },
      withdrawalRate: {
        intro: '7nDVZLSFxZcdHWmuSqzo6o',
        guide: '79KDyYdPfxwl7BceHPECCe',
        menu: '42bT4OaF5u9GXukOHXHnWz',
      },

      // This is never show because it is displayed only on detail screen,
      // but it is here to keep the code simple and uniform. Resusing content
      // from above.
      rewardRiskRatio: {
        intro: '7nDVZLSFxZcdHWmuSqzo6o',
        guide: '79KDyYdPfxwl7BceHPECCe',
      },
    },
    help: '1fLMnCI77Mz0WzvOaDTu2T',
  })
