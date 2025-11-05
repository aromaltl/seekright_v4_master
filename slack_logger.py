import datetime
import requests
import json
import config

arguments = config.getArgs()


class SlackLogger:
    def __init__(self):
        self.webhook_url = "https://hooks.slack.com/services/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"

    def postMessage(self, video_name, algo_type, text, status):
        current_time = datetime.datetime.now()
        slack_message_body = f"{video_name} is processed\n Algorithm:{algo_type}" \
                             + f" \n video_date :{current_time}" + f"\n message:{text}"
        site_name = arguments["site_name"].lower()
        
        print(site_name)
        if status == 1:
            _channel = f"{site_name}_video_upload"
        else:
            _channel = f"{site_name}_video_exception"

        self.postToSlack(_channel, slack_message_body)

    def postToSlack(self, channel, body):

        message_dict = {
            'channel': channel,
            'text': body
        }

        try:
            response = requests.post(self.webhook_url, data=json.dumps(message_dict))
            # response.raise_for_status()
        except requests.exceptions.RequestException as err:
            print(f'An error occurred while pushing the exception to Slack: {err}')


# if __name__ == "__main__":
#     obj = SlackLogger()
#     obj.postToSlack('adtpl_video_upload','ADTPL')

