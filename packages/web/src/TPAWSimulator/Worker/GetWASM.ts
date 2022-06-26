const _get =async()=> await import('@tpaw/simulator')
let _singletonPerWorker: Awaited<ReturnType<typeof _get>> | null = null

export const getWASM = async () => {
  if(!_singletonPerWorker){
    _singletonPerWorker = await _get()
  }
  return _singletonPerWorker
}
