module.exports = {
  apps: [
    {
      name: 'tpaw-dev-web',
      cwd:'./packages/web',
      script: 'npm',
      args: 'run dev',
    },
    {
      name: 'tpaw-dev-server',
      cwd:'./packages/server',
      script: 'npm',
      args: 'run start:dev',
    },
    {
      name: 'tpaw-dev-printserver',
      cwd:'./packages/printserver',
      script: 'npm',
      args: 'run start:dev',
    },
  ],
}
