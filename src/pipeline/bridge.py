import asyncio
import threading

class StreamlitAsyncBridge:
    @staticmethod
    def run_async_generator(async_gen):
        """
        Safely bridges an asynchronous generator into a synchronous Streamlit environment
        without causing event loop collision errors.
        """
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever)
        thread.start()

        try:
            while True:
                future = asyncio.run_coroutine_threadsafe(async_gen.__anext__(), loop)
                try:
                    yield future.result()
                except StopAsyncIteration:
                    break
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
