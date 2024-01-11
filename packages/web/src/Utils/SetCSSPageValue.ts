let _style = null as null | HTMLStyleElement
export const setCSSPageValue = (value: string) => {
  if (!_style) {
    _style = document.createElement('style')
    document.head.appendChild(_style)
  }
  _style.innerHTML = `@page {
    ${value}
  }`
}
