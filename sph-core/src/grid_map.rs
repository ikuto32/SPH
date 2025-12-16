/// A simple grid-based spatial hash used for neighborhood queries.
#[derive(Debug, Clone)]
pub struct GridMap {
    _width: i32,
    _height: i32,
    chunk_range: f32,
    num_chunk_width: i32,
    num_chunk_height: i32,
    chunks: Vec<Vec<usize>>,
}

impl GridMap {
    pub fn new(width: f32, height: f32, radius: f32) -> Self {
        let width_i = width.floor() as i32;
        let height_i = height.floor() as i32;
        let num_chunk_width = (width / radius).ceil() as i32 + 1;
        let num_chunk_height = (height / radius).ceil() as i32 + 1;
        let num_chunk = (num_chunk_width * num_chunk_height) as usize;
        Self {
            _width: width_i,
            _height: height_i,
            chunk_range: radius,
            num_chunk_width,
            num_chunk_height,
            chunks: vec![Vec::new(); num_chunk],
        }
    }

    fn chunk_index(&self, chunk_x: i32, chunk_y: i32) -> Option<usize> {
        if chunk_x < 0
            || chunk_x >= self.num_chunk_width
            || chunk_y < 0
            || chunk_y >= self.num_chunk_height
        {
            return None;
        }
        let idx = (chunk_y * self.num_chunk_width + chunk_x) as usize;
        Some(idx)
    }

    pub fn register_target(&mut self, target: usize, x: f32, y: f32) {
        let chunk_x = (x / self.chunk_range) as i32;
        let chunk_y = (y / self.chunk_range) as i32;
        if let Some(idx) = self.chunk_index(chunk_x, chunk_y) {
            self.chunks[idx].push(target);
        }
    }

    pub fn unregister_all(&mut self) {
        for chunk in &mut self.chunks {
            chunk.clear();
        }
    }

    pub fn find_neighborhood(&self, x: f32, y: f32, radius: f32) -> Vec<usize> {
        let center_chunk_x = (x / self.chunk_range) as i32;
        let center_chunk_y = (y / self.chunk_range) as i32;
        let radius_chunk = (radius / self.chunk_range).ceil() as i32;
        let min_chunk_x = center_chunk_x - radius_chunk;
        let max_chunk_x = center_chunk_x + radius_chunk;
        let min_chunk_y = center_chunk_y - radius_chunk;
        let max_chunk_y = center_chunk_y + radius_chunk;

        let mut out = Vec::new();
        for x1 in min_chunk_x..=max_chunk_x {
            for y1 in min_chunk_y..=max_chunk_y {
                if let Some(idx) = self.chunk_index(x1, y1) {
                    out.extend(self.chunks[idx].iter().copied());
                }
            }
        }
        out
    }
}
