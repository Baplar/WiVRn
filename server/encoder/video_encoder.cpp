/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

// Include first because of incompatibility between Eigen and X includes
#include "driver/wivrn_session.h"

#include "video_encoder.h"

#include "encoder_settings.h"
#include "os/os_time.h"
#include "util/u_logging.h"
#include "wivrn_config.h"

#include <string>

#if WIVRN_USE_NVENC
#include "video_encoder_nvenc.h"
#endif
#if WIVRN_USE_VAAPI
#include "ffmpeg/video_encoder_va.h"
#endif
#if WIVRN_USE_X264
#include "video_encoder_x264.h"
#endif
#if WIVRN_USE_VULKAN_ENCODE
#include "video_encoder_vulkan_h264.h"
// #include "video_encoder_vulkan_h265.h"
#endif

namespace wivrn
{

VideoEncoder::sender::sender() :
        thread([this](std::stop_token t) {
	        while (not t.stop_requested())
	        {
		        data * d = nullptr;
		        {
			        std::unique_lock lock(mutex);
			        if (pending.empty())
				        cv.wait_for(lock, std::chrono::milliseconds(100));
			        else
				        d = &pending.front();
		        }
		        if (d and not d->span.empty())
		        {
			        d->encoder->SendData(d->span, true);
			        std::unique_lock lock(mutex);
			        pending.pop_front();
			        cv.notify_all();
		        }
	        }
	        std::unique_lock lock(mutex);
	        pending.clear();
	        cv.notify_all();
        })
{
}

void VideoEncoder::sender::push(data && d)
{
	std::unique_lock lock(mutex);
	pending.push_back(std::move(d));
	cv.notify_all();
}

void VideoEncoder::sender::wait_idle(VideoEncoder * encoder)
{
	std::unique_lock lock(mutex);
	while (std::ranges::any_of(pending, [=](auto & data) { return data.encoder == encoder; }))
		cv.wait_for(lock, std::chrono::milliseconds(100));
}

std::shared_ptr<VideoEncoder::sender> VideoEncoder::sender::get()
{
	static std::weak_ptr<VideoEncoder::sender> instance;
	static std::mutex m;
	std::unique_lock lock(m);
	auto s = instance.lock();
	if (s)
		return s;
	s.reset(new VideoEncoder::sender());
	instance = s;
	return s;
}

std::unique_ptr<VideoEncoder> VideoEncoder::Create(
        wivrn_vk_bundle & wivrn_vk,
        encoder_settings & settings,
        uint8_t stream_idx,
        int input_width,
        int input_height,
        float fps)
{
	using namespace std::string_literals;
	std::unique_ptr<VideoEncoder> res;
	if (settings.encoder_name == encoder_vulkan)
	{
#if WIVRN_USE_VULKAN_ENCODE
		switch (settings.codec)
		{
			case video_codec::h264:
				res = video_encoder_vulkan_h264::create(wivrn_vk, settings, fps);
				break;
			case video_codec::h265:
				throw std::runtime_error("h265 not supported for vulkan video encode");
				// res = video_encoder_vulkan_h265::create(wivrn_vk, settings, fps);
				// break;
			case video_codec::av1:
				throw std::runtime_error("av1 not supported for vulkan video encode");
		}
#else
		throw std::runtime_error("Vulkan video encode not enabled");
#endif
	}
	if (settings.encoder_name == encoder_x264)
	{
#if WIVRN_USE_X264
		res = std::make_unique<VideoEncoderX264>(wivrn_vk, settings, fps);
#else
		throw std::runtime_error("x264 encoder not enabled");
#endif
	}
	if (settings.encoder_name == encoder_nvenc)
	{
#if WIVRN_USE_NVENC
		res = std::make_unique<VideoEncoderNvenc>(wivrn_vk, settings, fps);
#else
		throw std::runtime_error("nvenc support not enabled");
#endif
	}
	if (settings.encoder_name == encoder_vaapi)
	{
#if WIVRN_USE_VAAPI
		res = std::make_unique<video_encoder_va>(wivrn_vk, settings, fps);
#else
		throw std::runtime_error("vaapi support not enabled");
#endif
	}
	if (not res)
		throw std::runtime_error("Failed to create encoder " + settings.encoder_name);
	res->stream_idx = stream_idx;

	auto wivrn_dump_video = std::getenv("WIVRN_DUMP_VIDEO");
	if (wivrn_dump_video)
	{
		std::string file(wivrn_dump_video);
		file += "-" + std::to_string(stream_idx);
		switch (settings.codec)
		{
			case h264:
				file += ".h264";
				break;
			case h265:
				file += ".h265";
				break;
			case av1:
				file += ".av1";
				break;
		}
		res->video_dump.open(file);
	}
	return res;
}

#if WIVRN_USE_VULKAN_ENCODE
std::pair<std::vector<vk::VideoProfileInfoKHR>, vk::ImageUsageFlags> VideoEncoder::get_create_image_info(const std::vector<encoder_settings> & settings)
{
	std::pair<std::vector<vk::VideoProfileInfoKHR>, vk::ImageUsageFlags> result;
	for (const auto & item: settings)
	{
		if (item.encoder_name == encoder_vulkan)
		{
			result.second |= vk::ImageUsageFlagBits::eVideoEncodeSrcKHR;
			switch (item.codec)
			{
				case h264:
					result.first.push_back(video_encoder_vulkan_h264::video_profile_info.get());
					break;
				case h265:
					// result.first.push_back(video_encoder_vulkan_h265::video_profile_info.get());
					break;
				case av1:
					throw std::runtime_error("av1 not supported for vulkan video encode");
			}
		}
	}
	return result;
}
#endif

static const uint64_t idr_throttle = 100;

VideoEncoder::VideoEncoder(bool async_send) :
        last_idr_frame(-idr_throttle),
        shared_sender(async_send ? sender::get() : nullptr)
{}

VideoEncoder::~VideoEncoder()
{
	if (shared_sender)
		shared_sender->wait_idle(this);
}

void VideoEncoder::SyncNeeded()
{
	sync_needed = true;
}

void VideoEncoder::PresentImage(vk::Image y_cbcr, vk::raii::CommandBuffer & cmd_buf)
{
	// Wait for encoder to be done
	busy[next_present].wait(true);

	busy[next_present] = true;
	present_image(y_cbcr, cmd_buf, next_present);
	next_present = (next_present + 1) % num_slots;
}

void VideoEncoder::Encode(wivrn_session & cnx,
                          const to_headset::video_stream_data_shard::view_info_t & view_info,
                          uint64_t frame_index)
{
	assert(busy[next_encode].load());
	if (shared_sender)
		shared_sender->wait_idle(this);
	this->cnx = &cnx;
	auto target_timestamp = std::chrono::steady_clock::time_point(std::chrono::nanoseconds(view_info.display_time));
	bool idr = sync_needed.exchange(false);
	// Throttle idr to prevent overloading the decoder
	if (idr and frame_index < last_idr_frame + idr_throttle)
	{
		U_LOG_D("Throttle IDR: stream %d frame %ld", stream_idx, frame_index);
		sync_needed = true;
		idr = false;
	}
	if (idr)
		last_idr_frame = frame_index;
	const char * extra = idr ? ",idr" : ",p";
	clock = cnx.get_offset();

	timing_info = {
	        .encode_begin = clock.to_headset(os_monotonic_get_ns()),
	};
	cnx.dump_time("encode_begin", frame_index, os_monotonic_get_ns(), stream_idx, extra);

	// Prepare the video shard template
	shard.stream_item_idx = stream_idx;
	shard.frame_idx = frame_index;
	shard.shard_idx = 0;
	shard.view_info = view_info;
	shard.timing_info.reset();

	std::exception_ptr ex;
	try
	{
		auto data = encode(idr, target_timestamp, next_encode);
		cnx.dump_time("encode_end", frame_index, os_monotonic_get_ns(), stream_idx, extra);
		if (data)
		{
			timing_info.encode_end = clock.to_headset(os_monotonic_get_ns());
			assert(shared_sender);
			shared_sender->push(std::move(*data));
		}
	}
	catch (...)
	{
		ex = std::current_exception();
	}
	busy[next_encode] = false;
	busy[next_encode].notify_all();
	next_encode = (next_encode + 1) % num_slots;
	if (ex)
		std::rethrow_exception(ex);
}

void VideoEncoder::SendData(std::span<uint8_t> data, bool end_of_frame)
{
	std::lock_guard lock(mutex);
	if (end_of_frame)
	{
		timing_info.send_end = clock.to_headset(os_monotonic_get_ns());
		if (not timing_info.encode_end)
			timing_info.encode_end = timing_info.send_end;
	}
	if (video_dump)
		video_dump.write((char *)data.data(), data.size());
	if (shard.shard_idx == 0)
	{
		cnx->dump_time("send_begin", shard.frame_idx, os_monotonic_get_ns(), stream_idx);
		timing_info.send_begin = clock.to_headset(os_monotonic_get_ns());
	}

	shard.flags = to_headset::video_stream_data_shard::start_of_slice;
	auto begin = data.begin();
	auto end = data.end();
	while (begin != end)
	{
		const size_t view_info_size = sizeof(to_headset::video_stream_data_shard::view_info_t);
		const size_t max_payload_size = to_headset::video_stream_data_shard::max_payload_size - (shard.view_info ? view_info_size : 0);
		auto next = std::min(end, begin + max_payload_size);
		if (next == end)
		{
			shard.flags |= to_headset::video_stream_data_shard::end_of_slice;
			if (end_of_frame)
			{
				shard.flags |= to_headset::video_stream_data_shard::end_of_frame;
				shard.timing_info = timing_info;
			}
		}
		shard.payload = {begin, next};
		try
		{
			cnx->send_stream(shard);
		}
		catch (...)
		{
			// Ignore network errors
		}
		++shard.shard_idx;
		shard.flags = 0;
		shard.view_info.reset();
		begin = next;
	}
	if (end_of_frame)
		cnx->dump_time("send_end", shard.frame_idx, os_monotonic_get_ns(), stream_idx);
}

} // namespace wivrn
