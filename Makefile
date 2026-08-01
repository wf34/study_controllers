.PHONY: sync run

sync:
	uv sync
	uv run python tools/clear_execstack.py

run:
	uv run python controllers/3d.py \
	    --replay_browser chromium \
	    --select_controller stiffness \
	    --turns 3 \
	    $(ARGS)
