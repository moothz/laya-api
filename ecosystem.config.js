module.exports = {
  apps: [
    {
      name: "laya-gpu",
      script: "./start-gpu.sh",
      cwd: "/mnt/data/services/laya-api",
      interpreter: "bash",
      exec_mode: "fork",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
      env: {
        PORT: 8002,
        NODE_ENV: "production"
      }
    },
    {
      name: "laya-cpu",
      script: "./start-cpu.sh",
      cwd: "/mnt/data/services/laya-api",
      interpreter: "bash",
      exec_mode: "fork",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
      env: {
        PORT: 8005,
        WORKERS: 4,
        NODE_ENV: "production"
      }
    }
  ]
};
