module.exports = {
  apps: [
    {
      name: 'web',
      cwd:'./packages/web',
      script: 'npm',
      args: 'run dev',
    },
    {
      name: 'server',
      cwd:'./packages/server',
      script: 'npm',
      args: 'run start:dev',
    },
    {
      name: 'printserver',
      cwd:'./packages/printserver',
      script: 'npm',
      args: 'run start:dev',
    },
  ],
}
