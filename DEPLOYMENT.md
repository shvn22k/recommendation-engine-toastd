# Deployment Guide

This guide walks through deploying the Toastd recommendation system using:
- **Vercel** - Frontend (static HTML)
- **Fly.io** - Backend (FastAPI + ML model)
- **Upstash** - Redis cache (serverless)

---

## Step 1: Set Up Upstash Redis

1. Go to [upstash.com](https://upstash.com) and create an account

2. Click **Create Database**
   - Name: `toastd-cache`
   - Region: Choose closest to your Fly.io region (e.g., `us-west-1` for sjc)
   - Type: Regional (free tier)

3. After creation, go to the database details and copy the **Redis URL**
   - It looks like: `rediss://default:xxxxx@xxxx.upstash.io:6379`
   - Note: It's `rediss://` (with double s) for SSL

4. Save this URL - you'll need it for Fly.io

---

## Step 2: Deploy Backend on Fly.io

### 2.1 Install Fly CLI

**Windows (PowerShell)**:
```powershell
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"
```

**Or download from**: https://fly.io/docs/hands-on/install-flyctl/

### 2.2 Login to Fly.io

```bash
fly auth login
```

This opens a browser to authenticate.

### 2.3 Launch the App

From your project root directory:

```bash
fly launch
```

When prompted:
- App name: `toastd-api` (or your choice)
- Region: `sjc` (San Jose) or closest to you
- PostgreSQL: **No**
- Redis: **No** (we're using Upstash)
- Deploy now: **No** (we need to set secrets first)

### 2.4 Set Environment Variables (Secrets)

```bash
fly secrets set GEMINI_API_KEY=your-api-key-here
fly secrets set REDIS_URL=rediss://default:xxxxx@xxxx.upstash.io:6379
```

### 2.5 Deploy

```bash
fly deploy
```

First deploy takes ~5-10 minutes (downloading ML model dependencies).

### 2.6 Check Deployment

```bash
fly status
```

Your app URL will be:
```
https://toastd-api.fly.dev
```

Test it:
```bash
curl https://toastd-api.fly.dev/health
```

### 2.7 View Logs (if needed)

```bash
fly logs
```

---

## Step 3: Deploy Frontend on Vercel

### 3.1 Update API URL in Frontend

Update `frontend/index.html` (~line 325):

```javascript
const API_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'https://toastd-api.fly.dev';  // ← Your Fly.io URL
```

Commit and push this change.

### 3.2 Deploy to Vercel

**Option A: Via Vercel CLI**

```bash
npm i -g vercel
cd frontend
vercel
```

**Option B: Via Vercel Dashboard**

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub

2. Click **Add New** → **Project**

3. Import your `toastd` repository

4. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `frontend`
   - **Build Command**: (leave empty)
   - **Output Directory**: (leave empty)

5. Click **Deploy**

### 3.3 Get Your Frontend URL

Vercel gives you a URL like:
```
https://toastd.vercel.app
```

---

## Step 4: Test Everything

1. Open your Vercel frontend URL
2. Try a search query
3. Check browser console for any errors

---

## Environment Variables Summary

### Fly.io (Backend)

Set via `fly secrets set`:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Google AI Studio API key |
| `REDIS_URL` | Upstash Redis connection URL |

### Vercel (Frontend)

No environment variables needed - API URL is in the code.

---

## Useful Fly.io Commands

```bash
# Check app status
fly status

# View logs
fly logs

# SSH into the machine
fly ssh console

# Scale memory (if needed for ML model)
fly scale memory 1024

# Restart the app
fly apps restart toastd-api

# Open app in browser
fly open
```

---

## Troubleshooting

### Build fails on Fly.io

- Check `fly logs` for errors
- ML dependencies are large, might need more memory:
  ```bash
  fly scale memory 1024
  ```

### "GEMINI_API_KEY not found"

- Check secrets are set:
  ```bash
  fly secrets list
  ```
- Redeploy after setting secrets:
  ```bash
  fly deploy
  ```

### Redis connection failed

- Verify REDIS_URL is correct (should start with `rediss://` for Upstash)
- Check Upstash dashboard for connection issues

### CORS errors in browser

- The backend allows all origins by default
- If you restricted it, add your Vercel domain to `allow_origins`

### Slow first request (cold start)

- Normal! First request after idle loads the ML model (~10-30 seconds)
- Fly.io auto-stops idle machines by default
- To keep warm, set `min_machines_running = 1` in fly.toml (uses more resources)

---

## Costs

### Free Tier Limits

| Service | Free Tier |
|---------|-----------|
| Vercel | 100GB bandwidth/month |
| Fly.io | 3 shared-cpu-1x VMs, 256MB each |
| Upstash | 10,000 commands/day |

**Note**: The ML model needs ~400-500MB RAM. Fly.io free tier gives 256MB per VM which might be tight. You may need to:
- Scale to 512MB: `fly scale memory 512` (still within free tier usually)
- Or use their $5/month hobby plan

---

## Updating the Deployment

### Backend changes

```bash
git push origin main
fly deploy
```

### Frontend changes

Push to GitHub → Vercel auto-deploys

### Updating embeddings

If you add new products:
1. Run `prod_embeddings.py` locally
2. Replace `product_data.pkl` in the repo
3. Push and redeploy:
   ```bash
   git push origin main
   fly deploy
   ```
