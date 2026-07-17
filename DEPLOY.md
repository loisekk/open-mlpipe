# Deploy to Render

## Option 1: Manual Deploy (Recommended)

1. Go to https://render.com
2. Sign up / Login with GitHub
3. Click **"New +"** → **"Static Site"**
4. Connect GitHub repo: `loisekk/open-mlpipe`
5. Configure:
   - **Name:** `open-mlpipe-website`
   - **Branch:** `main`
   - **Root Directory:** `website`
   - **Build Command:** (leave empty)
   - **Publish Directory:** `.`
6. Click **"Create Static Site"**
7. Wait 1-2 minutes for deployment
8. Your site will be live at: `https://open-mlpipe-website.onrender.com`

## Option 2: Quick Deploy

1. Go to https://render.com/deploy
2. Click **"Deploy to Render"**
3. Select your repo
4. Set Root Directory to `website`
5. Deploy

## What You Get

- ✅ Free SSL certificate
- ✅ Custom domain support
- ✅ Auto-deploy on push
- ✅ Global CDN
- ✅ Fast loading

## After Deploy

Your site will be live at:
```
https://open-mlpipe-website.onrender.com
```

You can add a custom domain later in Render settings.
