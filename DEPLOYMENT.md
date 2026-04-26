# Deployment Guide for VOGEN

## Hugging Face Spaces Deployment

1. **Create a new Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `vogen` (or your choice)
   - SDK: `Docker`
   - Visibility: Public
   - Click Create

2. **Push your code**:
   ```bash
   # In your local repo
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/vogen
   git push space main
   ```

3. **The Space will automatically build** using the `Dockerfile` in your repo.

4. **Get the Space URL**: `https://huggingface.co/spaces/YOUR_USERNAME/vogen`

## Blog Post on Hugging Face

1. **Go to HF Blog**: https://huggingface.co/blog

2. **Create a new post**:
   - Click "Write a blog post"
   - Title: "Training LLMs to Style Outfits: VOGEN Environment"
   - Copy content from `blog.md`
   - Add images: Upload `results/reward_curve.png`
   - Publish

3. **Get the blog URL**: It will be something like `https://huggingface.co/blog/training-llms-style-outfits-vogen`

## Update README

Once you have the URLs:

1. Edit `README.md`
2. Replace `https://huggingface.co/blog/...` with actual blog URL
3. Replace `https://huggingface.co/spaces/...` with actual Space URL
4. Commit and push

## Alternative: YouTube Video

If you prefer a video:

1. Create a <2 minute video explaining:
   - Problem: Why fashion styling matters for LLMs
   - Environment: Show observations/actions/rewards
   - Results: Show the training curves
   - Demo: Quick interaction with the environment

2. Upload to YouTube

3. Update README with YouTube link

## Final Checklist

- [ ] HF Space deployed and accessible
- [ ] Blog post published or video uploaded
- [ ] README updated with links
- [ ] All links working
- [ ] Test the Space works