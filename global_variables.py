class variables:
    def __init__(self):
        ### metadata_samecam.py####
        self.fps = None
        self.video_frame_count = None
        self.video_name = None
        self.video_path = None
        self.video_id = None
        ############################
        self.algorithm = None #<algorithm>.py or run.py for reupload
        self.gantry_prev_frame = 0
        self.gantry_prev_x_y = (None, None)


variable_obj = variables()
